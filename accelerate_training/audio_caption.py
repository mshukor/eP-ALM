import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


import os, sys
sys.path.append(os.path.abspath('.')) # ~/ep-alm

from models.epalm import ePALM
from models.utils import freeze_whole_model, unfreeze_parameters, print_trainable_params_percentage
from models.utils import filter_state, filter_msg, exclude_list

from transformers import AutoTokenizer

import utils

from dataset.audio_caption import get_loader 
from scheduler import create_scheduler
from optim import create_optimizer
 

from accelerate import Accelerator

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, accelerator=None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ", accelerator=accelerator)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    lm_loss_weight = config.get('lm_loss_weight', 1)

    append_eos_token = config.get('append_eos_token', False)
    eos_token = tokenizer.eos_token

    config_optim = utils.AttrDict(config['optimizer'])
    prompt_lr = config_optim.prompt_lr if hasattr(config_optim, 'prompt_lr') else None


    if prompt_lr is not None:
        metric_logger.add_meter('prompt_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))


    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch["images"].to(device,non_blocking=True)

        text = batch["sent"]

        if append_eos_token:
            text = [t.replace(eos_token, '') + eos_token for t in text]





        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device) 


        targets = text_input.input_ids.masked_fill(text_input.input_ids == tokenizer.pad_token_id, -100)
        

        answer_output = model(image=image, 
                              text=text_input, 
                              labels = targets,
                              return_dict = True,   
                              mode='train',
                              reduction='none',
                             )      
        
        loss = answer_output.loss         
        loss = loss.sum()/image.size(0)
        loss = loss*lm_loss_weight
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if prompt_lr is not None:
            metric_logger.update(prompt_lr=optimizer.param_groups[1]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 

            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    accelerator.print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 




@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, accelerator=None):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate Caption test result:'
    print_freq = 50
    
        
    predictions = []
    targets = []



    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token

    num_beams = config.get('num_beams', 1)
    do_sample = config.get('do_sample', True)
    accelerator.print("num_beams", num_beams, "do_sample", do_sample)

    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        
        image = batch["images"].to(device,non_blocking=True)
        text = ['' for q in image]  

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device) 

        out = model(image=image, text=text_input, mode='generate', return_dict=True, max_length=30, 
        do_sample=do_sample, num_beams=num_beams)
        out_decode = []
        for i, o in enumerate(out):
            try:

                res = tokenizer.decode(o)
                response = res.split('</s>')[1].replace(pad_token, '').replace('</s>', '').replace(eos_token, '') # skip_special_tokens=True
            except TypeError:
                accelerator.print(o)
                response = ' '
            out_decode.append(response)


        predictions.extend(out_decode)

        if 'targets' in batch:
            targets.extend(batch['targets'])




    evaluator = data_loader.evaluator
    eval_results = evaluator.evaluate(predictions, targets)


    wandb_log_dict = {}

    for score_name, score in eval_results.items():
        wandb_log_dict[f'Valid/{score_name}'] = score


    accelerator.print(wandb_log_dict)

    return wandb_log_dict



def main(args, config):
    if 'XDG_CACHE_HOME' in os.environ:
        os.environ['TORCH_HOME'] = os.environ['XDG_CACHE_HOME']+'/torch'
    else:
        os.environ['TORCH_HOME'] = '~/.cache/torch'
    args.distributed = False
    accelerator = Accelerator()    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    accelerator.print(args, config)


    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False, local_files_only=True)



    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()     
    else:
        num_tasks = None
        global_rank = None


    #########
    num_workers = config.get('num_workers', 4)
    train_topk = config.get('train_topk', -1)
    valid_topk = config.get('valid_topk', -1)
    data_dir = args.data_dir

    args.image_size = config.get('image_res', 224)
    args.use_data_augmentation = True 

    black_image = config.get('black_image', False)

    accelerator.print("black image:", black_image)

    # audio 
    args.melbins = config.get('melbins', 128)
    args.target_length = config.get('target_length', 1024)
    args.num_tries = config.get('num_tries', 1)

    args.skip_norm = config.get('skip_norm', True)
    args.norm_mean = config.get('norm_mean', None)
    args.norm_std = config.get('norm_std', None)
    args.noise = config.get('noise', False)

    args.freqm_p = config.get('freqm_p', 48)
    args.timem_p = config.get('timem_p', 192)


    train_split = config.get('train_split', 'train') 
    val_split = config.get('val_split', 'val') 
    test_split = config.get('test_split', 'test') 


    train_loader = get_loader(
        args,
        split=train_split, mode='train', batch_size=config['batch_size_train'],
        distributed=args.distributed,
        workers=num_workers,
        topk=train_topk,
        data_dir=data_dir,
        local_rank=global_rank, world_size=num_tasks, verbose=True, black_image=black_image
    )

    accelerator.print('# len train loader:', len(train_loader))
    accelerator.print(f'Building val loader')
    val_loader = get_loader(
        args,
        split=val_split, mode='val', batch_size=config['batch_size_test'],
        distributed=False, 
        workers=4,
        topk=valid_topk,data_dir=data_dir,
        local_rank=global_rank, world_size=num_tasks, verbose=True, black_image=black_image
    )
    accelerator.print('# len val loader:', len(val_loader))

    accelerator.print(f'Building test loader')
    test_loader = get_loader(
        args,
        split=test_split, mode='val', batch_size=config['batch_size_test'],
        distributed=False, 
        workers=4,
        topk=valid_topk,data_dir=data_dir,
        local_rank=global_rank, world_size=num_tasks, verbose=True
    )


    accelerator.print('# len test loader:', len(test_loader))

    #### Model #### 
    accelerator.print("Creating model")
    
    start_layer_idx = config.get('start_layer_idx', 0)
    end_layer_idx = config.get('end_layer_idx', 0)

    vision_model_name = config.get('vision_model_name', args.vision_model)

    model = ePALM(opt_model_name = args.text_model, 
                   vision_model_name = vision_model_name, 
                   use_vis_prefix = True, 
                   start_layer_idx = start_layer_idx, 
                   end_layer_idx = end_layer_idx, 
                   return_hidden_state_vision = True, 
                   config=config,
                   low_cpu=args.low_cpu
    )
    
        
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, config=config)

    if hasattr(arg_opt, 'prompt_lr') and arg_opt.prompt_lr is not None:
        accelerator.print('\tInitial other params params lr: %f' % optimizer.param_groups[0]['lr'])
        accelerator.print('\tInitial prompt params lr: %f' % optimizer.param_groups[1]['lr'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
         
    best_epoch = 0 
    best_valid = 0 
    
    if args.checkpoint:    

        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict,strict=False)  
        msg = filter_msg(msg, exclude_list)
        accelerator.print('load checkpoint from %s'%args.checkpoint)
        accelerator.print(msg)  

        if args.resume:
            model = model.to(device) 
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1  
            accelerator.print(checkpoint.keys())
            for p in optimizer.param_groups: # not necessay after torch 1.12.1
                p['capturable'] = True
        if 'best_valid' in checkpoint:
            best_valid = checkpoint['best_valid'] 
            best_epoch = checkpoint['best_epoch'] 
            accelerator.print("load best valid {} at epoch {}".format(best_valid, best_epoch))

    
    freeze_whole_model(model)
    unfreeze_parameters(model, config)
    print_trainable_params_percentage(model)
    
    val_evaluator = val_loader.evaluator
    test_evaluator = test_loader.evaluator
    task  = val_loader.task
    device = accelerator.device

    model, optimizer, train_loader, val_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
    )
    model = model.to(device) 

    test_loader.evaluator = test_evaluator
    val_loader.evaluator = val_evaluator

    test_loader.task = task
    val_loader.task = task
    
    
    accelerator.print("Start training")
    start_time = time.time()


    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, 
                                lr_scheduler, config, accelerator=accelerator)  

        if args.evaluate:
            break
            

        valid_results = evaluation(model, val_loader, tokenizer, device, config, accelerator=accelerator) 

        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                    
            ## avoid memory issue with accelerator.get_state_dict
            state_dict = accelerator.unwrap_model(model)
            state_dict = state_dict.state_dict()
            state_dict = filter_state(state_dict, exclude_list) # filter_state(model_without_ddp.state_dict(), exclude_list)
            if state_dict is not None:
                for k in state_dict:
                    if state_dict[k].dtype == torch.float16:
                        state_dict[k] = state_dict[k].float()


            save_obj = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_valid': best_valid,
                'best_epoch': best_epoch,
            }


            if args.save_best:
                valid_score = valid_results['Valid/CIDEr']

                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    accelerator.print("Save best epoch:", best_epoch)

                    save_obj['best_valid'] = best_valid
                    save_obj['best_epoch'] = best_epoch

                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))  


        dist.barrier()   
    




    ### test best model
    if not args.evaluate:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_best.pth'), map_location='cpu') 
        state_dict = checkpoint['model']   
        msg = model.module.load_state_dict(state_dict,strict=False)  
        msg = filter_msg(msg, exclude_list)
        accelerator.print('load checkpoint for test from %s'%os.path.join(args.output_dir, 'checkpoint_best.pth'))
        accelerator.print(msg)
        print("best_epoch", checkpoint['best_epoch'], "best_valid", checkpoint['best_valid'])
    print("best_epoch", best_epoch, "best_valid", best_valid)
    vqa_result = evaluation(model, test_loader, tokenizer, device, config, accelerator=accelerator)    

                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_model', default='facebook/opt-350m')
    parser.add_argument('--vision_model', default='vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    parser.add_argument('--data_dir', default='/data/mshukor/data')   
    parser.add_argument('--resume', action='store_true')    

    parser.add_argument('--save_best', action='store_true') 
    
    parser.add_argument('--image_dir', default='/data/mshukor/data')   

    
    parser.add_argument('--low_cpu', action='store_true') 
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)