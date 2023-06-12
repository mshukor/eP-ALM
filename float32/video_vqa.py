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
 

from models.epalm import ePALM
from models.utils import freeze_whole_model, unfreeze_parameters, print_trainable_params_percentage



from transformers import AutoTokenizer


import utils

from dataset.video_vqa import get_loader 

from scheduler import create_scheduler
from optim import create_optimizer
 


from tqdm import tqdm 


from models.utils import filter_state, filter_msg, exclude_list


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()  
    

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    config_optim = utils.AttrDict(config['optimizer'])
    prompt_lr = config_optim.prompt_lr if hasattr(config_optim, 'prompt_lr') else None
    connector_lr = config_optim.connector_lr if hasattr(config_optim, 'connector_lr') else None
    vis_lr = config_optim.vis_lr if hasattr(config_optim, 'vis_lr') else None
    text_lr = config_optim.text_lr if hasattr(config_optim, 'text_lr') else None

    print(vis_lr, text_lr, connector_lr, len(optimizer.param_groups))
    if prompt_lr is not None:
        metric_logger.add_meter('prompt_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))



    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    lm_loss_weight = config.get('lm_loss_weight', 1)
    special_answer_token = config.get('special_answer_token', None)

    special_eo_answer_token = config.get('special_eo_answer_token', None)




    eos_token = tokenizer.eos_token if special_eo_answer_token is None else special_eo_answer_token

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch['images'].to(device,non_blocking=True)

        question = batch['sent']

        answer = batch['answers']

        
        questions_answers = []


        if special_answer_token is not None:
            questions_answers += [question[i] + "?" + special_answer_token + answer[i].replace('[SEP]','') + eos_token for i in range(len(question))]  
        else:
            questions_answers += [question[i] + "</s>" + answer[i].replace('[SEP]','') +  eos_token for i in range(len(question))]  

        questions_answers_input = tokenizer(questions_answers, padding='longest', return_tensors="pt").to(device) 
        answer_targets = questions_answers_input.input_ids.masked_fill(questions_answers_input.input_ids == tokenizer.pad_token_id, -100)

        images = image

        

        answer_output = model(image=images, 
                              text=questions_answers_input, 
                              labels = answer_targets,
                              return_dict = True,   
                              mode='train',
                              reduction='none',
                             )      
        
        loss = answer_output.loss         
        loss = loss.sum()/image.size(0)
        loss = loss*lm_loss_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if prompt_lr is not None:
            metric_logger.update(prompt_lr=optimizer.param_groups[1]["lr"])

        if i % print_freq == 0:
            lrs = [g["lr"] for g in optimizer.param_groups]
            print(lrs)

        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            if scheduler is not None:
                scheduler.step(i//step_size) 

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 



@torch.no_grad()
def predict(model, loader, tokenizer, device, dump_path=None, verbose=False, distributed=False, special_answer_token=None, special_eo_answer_token=None):
    model.eval()
    eos_token = tokenizer.eos_token if special_eo_answer_token is None else special_eo_answer_token
    pad_token = tokenizer.pad_token

    with torch.no_grad():
        quesid2ans = {}
        if verbose:
            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
        for i, batch in enumerate(loader):


            image = batch['images'].to(device,non_blocking=True)

            question = batch['sent']

            question_id = batch['question_ids']

            if special_answer_token is not None:
                question = [q+'?'+special_answer_token for q in question]
            else:
                question = [q+eos_token for q in question]

            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device) 

            out = model(image=image, text=question_input, mode='generate', return_dict=True, max_length=30, do_sample=True)
            
            


            for ques_id, o in zip(question_id, out):
                o_list = o.tolist()
                try:
                    if special_answer_token is not None:
                        response = tokenizer.decode(o_list).split(special_answer_token)[1].replace(pad_token, '').replace('</s>', '').replace(eos_token, '') # skip_special_tokens=True
                    else:
                        response = tokenizer.decode(o_list).split('</s>')[2].replace(pad_token, '').replace('</s>', '').replace(eos_token, '') # skip_special_tokens=True
                except TypeError:
                    print(o_list)
                    response = ' '

                ques_id = ques_id          
                quesid2ans[ques_id] = response  


            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

    if distributed:
        dist.barrier()

    qid2ans_list = utils.all_gather(quesid2ans)
    if verbose:
        quesid2ans = {}
        for qid2ans in qid2ans_list:
            for k, v in qid2ans.items():
                quesid2ans[k] = v

        if dump_path is not None:
            evaluator = loader.evaluator
            evaluator.dump_result(quesid2ans, dump_path)

    return quesid2ans


  

def evaluate(model, data_loader, tokenizer, device, 
        distributed=False, special_answer_token=None, special_eo_answer_token=None):
    verbose = utils.is_main_process()


    quesid2ans = predict(model, data_loader, tokenizer, device, verbose=verbose, 
        distributed=distributed, special_answer_token=special_answer_token, special_eo_answer_token=special_eo_answer_token)

    evaluator = data_loader.evaluator

    acc_dict = {}
    topk_score = evaluator.evaluate(quesid2ans, normalize_answer=True)
    acc_dict['topk_score'] = topk_score
    return acc_dict








def main(args, config):

    os.environ['TORCH_HOME'] = os.environ['XDG_CACHE_HOME']+'/torch'

    utils.init_distributed_mode(args)    
    
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
    
    
    #### Dataset #### 
    
    print("Creating dataset")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()     
    else:
        num_tasks = None
        global_rank = None
    
    num_workers = config.get('num_workers', 4)
    train_topk = config.get('train_topk', -1)
    valid_topk = config.get('valid_topk', -1)
    data_dir = args.data_dir

    args.image_size = config.get('image_res', 224)
    args.use_data_augmentation = True 


    # video 
    args.num_frames = config.get('num_frames', 4)
    args.as_images = config.get('as_images', True)
    args.num_tries = config.get('num_tries', 1)
    args.sample_type = config.get('sample_type', 'rand')

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
        local_rank=global_rank, world_size=num_tasks, verbose=True
    )

    args.raw_label = False
    print('# len train loader:', len(train_loader))
    print(f'Building val loader')
    val_loader = get_loader(
        args,
        split=val_split, mode='val', batch_size=config['batch_size_test'],
        distributed=args.distributed, 
        workers=4,
        topk=valid_topk,data_dir=data_dir,
        local_rank=global_rank, world_size=num_tasks, verbose=True
    )
    print('# len val loader:', len(val_loader))

    print(f'Building test loader')
    test_loader = get_loader(
        args,
        split=test_split, mode='val', batch_size=config['batch_size_test'],
        distributed=args.distributed, 
        workers=4,
        topk=valid_topk,data_dir=data_dir,
        local_rank=global_rank, world_size=num_tasks, verbose=True
    )


    print('# len test loader:', len(test_loader))


    if args.submit:
        print(f'Building test submit loader ...')
        submit_test_loader = get_loader(
            args,
            split='test', mode='val', batch_size=config['batch_size_test'],
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=valid_topk, data_dir=data_dir,
            local_rank=global_rank, world_size=num_tasks, verbose=True
        )

    #### Model #### 
    print("Creating model")
    
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
    )
    
        
    model = model.to(device)   
    

    tokenizer_name = config.get('tokenizer_name', args.text_model)
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, local_files_only=True)
   
    

    special_answer_token = config.get('special_answer_token', None)
    special_eo_answer_token = config.get('special_eo_answer_token', None)


    if special_answer_token is not None:
        special_tokens_dict = {'additional_special_tokens': [special_answer_token]}
        if special_eo_answer_token is not None:
            special_tokens_dict['additional_special_tokens'] += [special_eo_answer_token]

        tokenizer.add_special_tokens(special_tokens_dict)
        print("Adding special token:", special_tokens_dict)
        print(tokenizer)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, config=config['optimizer'])

    if hasattr(arg_opt, 'prompt_lr') and arg_opt.prompt_lr is not None:
        print('\tInitial other params params lr: %f' % optimizer.param_groups[0]['lr'])
        print('\tInitial prompt params lr: %f' % optimizer.param_groups[1]['lr'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
         
    
    if args.checkpoint:    

        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']


        msg = model.load_state_dict(state_dict,strict=False)  
        msg = filter_msg(msg, exclude_list)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

        if 'best_valid' in checkpoint:
            print("load best valid {} at epoch {}".format(checkpoint['best_valid'] , checkpoint['best_epoch'] ))

        if args.resume:
            model = model.to(device) 
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1  
            print(checkpoint.keys())
            if 'best_valid' in checkpoint:
                best_valid = checkpoint['best_valid'] 
                best_epoch = checkpoint['best_epoch'] 
                print("load best valid {} at epoch {}".format(best_valid, best_epoch))

    freeze_whole_model(model)
    unfreeze_parameters(model, config)
    print_trainable_params_percentage(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    
    print("Start training")
    start_time = time.time()

    best_valid = 0.
    best_epoch = 0

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            if lr_scheduler is not None:
                lr_scheduler.step(epoch+warmup_steps)  
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  

        if args.evaluate:
            break

        score_dict = evaluate(model, val_loader, tokenizer, device, distributed=args.distributed, 
            special_answer_token=special_answer_token, special_eo_answer_token=special_eo_answer_token)
        print(score_dict)
        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                     
            if lr_scheduler is None:
                lr_scheduler_state_dict = {}
            else:
                lr_scheduler_state_dict = lr_scheduler.state_dict()    
            save_obj = {
                'model': filter_state(model_without_ddp.state_dict(), exclude_list),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_state_dict,
                'config': config,
                'epoch': epoch,
                'best_valid': best_valid,
                'best_epoch': best_epoch,
            }

            if args.save_best:
                valid_score = score_dict['topk_score'] * 100.
                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    
                    save_obj['best_valid'] = best_valid
                    save_obj['best_epoch'] = best_epoch

                    print("save best epoch:", best_epoch)
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  

            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))                  

        dist.barrier()   
    
    if lr_scheduler is None:
        lr_scheduler_state_dict = {}
    else:
        lr_scheduler_state_dict = lr_scheduler.state_dict()    
    save_obj = {
        'model': filter_state(model_without_ddp.state_dict(), exclude_list),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler_state_dict,
        'config': config,
        'epoch': epoch,
        'best_valid': best_valid,
        'best_epoch': best_epoch,
    }
    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))  

    verbose = utils.is_main_process()

    
    if not args.evaluate:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_best.pth'), map_location='cpu') 
        state_dict = checkpoint['model']   
        msg = model.module.load_state_dict(state_dict,strict=False)  
        msg = filter_msg(msg, exclude_list)
        print('load checkpoint for test from %s'%os.path.join(args.output_dir, 'checkpoint_best.pth'))
        print(msg)

    quesid2ans = predict(model, test_loader, tokenizer, device, verbose=verbose, 
        distributed=args.distributed, special_answer_token=special_answer_token, special_eo_answer_token=special_eo_answer_token)

    evaluator = test_loader.evaluator
    score_dict = evaluator.evaluate(quesid2ans, normalize_answer=True)

    print("Test accuracy:", score_dict)


    if args.submit:
        dump_path = os.path.join(args.output_dir, 'submit.json')
        predict(submit_test_loader, dump_path)


    if args.distributed:
        dist.barrier()
        exit()


                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

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

    parser.add_argument('--submit', action='store_true') 
    parser.add_argument('--save_best', action='store_true') 



    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)