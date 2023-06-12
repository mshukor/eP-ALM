import re

import torch
from torch import nn





def freeze_whole_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
        
def unfreeze_parameters(model, config):       
    # targets = '*.proj_*|*_proj*|*itm_head*|*queue*|*adapter*|*temp*|*.cls.*'

    targets = ['connector'] # lm_head
    if config.get('unfreeze_text_layer_norm', False):
    	targets = targets + ['self_attn_layer_norm', 'final_layer_norm']  
    
    if config.get('unfreeze_vision_layer_norm', False):
    	targets = targets + ['norm', 'norm1', 'norm2']  
        
    print('unfreeze targets:', targets)
    for n, p in model.named_parameters():
    	if any(t in n for t in targets):
        # if re.fullmatch(targets, n):
            p.requires_grad = True
            print(f"{n} is trainable...")


def print_trainable_params_percentage(model):


    orig_param_size = sum(p.numel() for p in model.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_size = count_parameters(model)

    percentage = trainable_size / orig_param_size * 100

    print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

    return percentage