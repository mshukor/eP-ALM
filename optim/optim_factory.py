""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True, config=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    customized_lr = config.get('customized_lr', False) 
    prompt_lr = config.get('prompt_lr', args.lr) 
    vis_lr = config.get('vis_lr', args.lr)  
    text_lr = config.get('text_lr', args.lr)  
    connector_lr = config.get('connector_lr', args.lr)  
    adapter_lr = config.get('adapter_lr', args.lr)  
    
    if customized_lr:
        parameters = []
        targets = ['connector', 'model_vision', 'model_text', 'prompt']
        # if prompt_lr is not None:
        #     targets.append('prompt')
        #     promt_params = [kv[1] for kv in model.named_parameters() if 'prompt' in kv[0].split('.')[0]]
        #     parameters.append({'params': promt_params, 'lr': prompt_lr, 'weight_decay': weight_decay})

        params = {'connector': [], 'model_vision': [], 'model_text': [], 'prompt': [], 'other': []}
        for kv in model.named_parameters():
            if 'connector' in kv[0].split('.')[0]:
                params['connector'].append(kv[1])
            elif 'model_vision' in kv[0].split('.')[0]:
                params['model_vision'].append(kv[1])
            elif 'model_text' in kv[0].split('.')[0]:
                params['model_text'].append(kv[1])
            elif 'prompt' in kv[0].split('.')[0]:
                params['prompt'].append(kv[1])
            else:
                params['other'].append(kv[1])

        # connector_params = [kv[1] for kv in model.named_parameters() if 'connector' in kv[0].split('.')[0]]
        parameters.append({'params': params['connector'], 'lr': connector_lr, 'weight_decay': weight_decay})
        print('connector', len(params['connector']))

        # model_vision_params = [kv[1] for kv in model.named_parameters() if 'model_vision' in kv[0].split('.')[0]]
        parameters.append({'params': params['model_vision'], 'lr': vis_lr, 'weight_decay': weight_decay})
        print('model_vision', len(params['model_vision']))

        # model_text_params = [kv[1] for kv in model.named_parameters() if 'model_text' in kv[0].split('.')[0]]
        parameters.append({'params': params['model_text'], 'lr': text_lr, 'weight_decay': weight_decay})
        print('model_text', len(params['model_text']))

        parameters.append({'params': params['prompt'], 'lr': prompt_lr, 'weight_decay': weight_decay})
        print('prompt_lr', len(params['prompt']))

        # other_params = [kv[1] for kv in model.named_parameters() if any(t in kv[0].split('.')[0] for t in targets)]
        parameters.append({'params': params['other']})
        print('other', len(params['other']))
        print(connector_lr, vis_lr, text_lr, prompt_lr, "optim")
        
    elif config.get('adapter_lr', False) and config.get('prompt_lr', False):
        adapter_params = [kv[1] for kv in model.named_parameters() if 'adapter' in kv[0]]
        promt_params = [kv[1] for kv in model.named_parameters() if 'prompt' in kv[0]]

        other_params = [kv[1] for kv in model.named_parameters() if 'adapter' not in kv[0] and 'prompt' not in kv[0]]
        parameters = [
        {'params': other_params}, 
        {'params': adapter_params, 'lr': adapter_lr, 'weight_decay': weight_decay},
        {'params': promt_params, 'lr': prompt_lr, 'weight_decay': weight_decay}
        ] 

    elif config.get('adapter_lr', False):
        adapter_params = [kv[1] for kv in model.named_parameters() if 'adapter' in kv[0]]
        other_params = [kv[1] for kv in model.named_parameters() if 'adapter' not in kv[0]]
        parameters = [
        {'params': other_params}, 
        {'params': adapter_params, 'lr': adapter_lr, 'weight_decay': weight_decay}
        ]      

    elif config.get('prompt_lr', False):
        promt_params = [kv[1] for kv in model.named_parameters() if 'prompt' in kv[0]]
        other_params = [kv[1] for kv in model.named_parameters() if 'prompt' not in kv[0]]
        parameters = [
        {'params': other_params}, 
        {'params': promt_params, 'lr': prompt_lr, 'weight_decay': weight_decay}
        ]
    elif weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
