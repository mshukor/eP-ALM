import re

import torch
from torch import nn


exclude_list = ['model_text', 'transformer', 'model_vision']

def filter_msg(msg, exclude_list):
    new_msg = []
    if len(msg) > 1:
        for k in msg[0]: # missing
            if not any([e in k for e in exclude_list]) or 'adapter' in k:
                new_msg.append(k)
        return new_msg

def filter_state(state, exclude_list):
    import collections
    new_tmp = collections.OrderedDict()
    for k, v in state.items():
        if not any([e in k for e in exclude_list]) or 'adapter' in k:
            new_tmp[k] = state[k]
    
    return new_tmp



def freeze_whole_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
        
def unfreeze_parameters(model, config):       
    # targets = '*.proj_*|*_proj*|*itm_head*|*queue*|*adapter*|*temp*|*.cls.*'

    targets = ['prompt'] # lm_head

    

    if not config.get('freeze_connector', False):
        targets = targets + ['connector']

    if config.get('unfreeze_text_layer_norm', False):
        targets = targets + ['self_attn_layer_norm', 'final_layer_norm']  
    
    if config.get('unfreeze_vision_layer_norm', False):
        targets = targets + ['norm', 'norm1', 'norm2']  
        
    if config.get('unfreeze_text_model', False):
        targets = targets + ['model_text']  

    if config.get('unfreeze_vision_model', False):
        targets = targets + ['model_vision']

    if config.get('use_adapters', False):
        targets = targets + ['adapter']

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



def shift_right(input_ids, decoder_start_token_id=2, pad_token_id=None):


    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids