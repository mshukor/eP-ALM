import re

import torch
from torch import nn
from torchvision import transforms


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from models.opt import OPTModel, OPTConfig, OPTForCausalLM
import models.vit 

from PIL import Image
import json 
import numpy as np



import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding

def rank_answer(model, image, question_input, answer_ids, answer_atts, k, tokenizer):

    num_ques = question_input.input_ids.size(0)
    start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token

    start_ids = torch.cat((question_input.input_ids,  start_ids), dim=1)
    attention_mask = torch.cat((question_input.attention_mask,  torch.ones((num_ques, 1)).to(question_input.attention_mask.device)), dim=1)
    
    start_input = {'input_ids': start_ids, 'attention_mask': attention_mask}
    start_input = BatchEncoding(start_input)
    
    
    
    start_output = model(image, start_input, return_dict = True, mode='evaluate')     
    
    logits = start_output.logits[:,-1,:] # first token's logit

    # topk_probs: top-k probability 
    # topk_ids: [num_question, k]        
    answer_first_token = answer_ids[:,1]
    prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
    topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 

    # answer input: [num_question*k, answer_len]                 
    input_ids = []
    input_atts = []
    for b, topk_id in enumerate(topk_ids):
        input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
        input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
    input_ids = torch.cat(input_ids,dim=0)  
    input_atts = torch.cat(input_atts,dim=0)  

    start_ids = tile(start_ids, 0, k)
    attention_mask = tile(attention_mask, 0, k)
    image = tile(image, 0, k)
    
    
        
    
    input_ids = torch.cat((start_ids, input_ids), dim=1) # include the  <s> ?
    input_atts = torch.cat((attention_mask, input_atts), dim=1)
        
    targets_ids = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)

    
    
    # repeat encoder's output for top-k answers


    inputs = {'input_ids': input_ids, 'attention_mask': input_atts}
    inputs = BatchEncoding(inputs)
    
    output = model(image, inputs, labels = targets_ids, return_dict = True, mode='train', reduction='none')                 

    answer_loss = output.loss 
    answer_loss = answer_loss.view(input_ids.size(0),-1)

    # topk_prob: first token probability

    topk_probs = topk_probs.view(-1,1)
    log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

    # re-calculate log probabilities for the answer sequences using chain rule
    log_probs_sum = log_probs.sum(1)
    log_probs_sum = log_probs_sum.view(num_ques,k)

    topk_probs = F.softmax(log_probs_sum, dim=-1)
    # get top-k after re-ranking
    topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
    topk_ids = torch.gather(topk_ids, 1, rerank_id)    

    return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    





class VisOPT(nn.Module):
    def __init__(self,                 
                 opt_model_name = 'facebook/opt-350m',
                 vision_model_name = 'vit_base_patch16_224',
                 use_vis_prefix = True,
                 start_layer_idx = 11,
                 end_layer_idx = 23,
                 return_hidden_state_vision = True,
                 injected_hidden_states = 1,
                 
                 ):
        super().__init__()
        print("Loading VisOPT ...")
        # text
        config_opt = AutoConfig.from_pretrained(opt_model_name)
        
        config_opt.use_vis_prefix = use_vis_prefix
        config_opt.start_layer_idx = start_layer_idx
        config_opt.end_layer_idx = end_layer_idx
            
        print(config_opt)
        print("Loading: ", opt_model_name)
        self.model_text = OPTForCausalLM.from_pretrained(opt_model_name, config=config_opt)
        
        # vision
        print("Loading: ", vision_model_name)
        vision_func = getattr(models.vit, vision_model_name)
        self.model_vision = vision_func(pretrained=True, return_hidden_state=return_hidden_state_vision)
        
        # connector
        self.injected_hidden_states = injected_hidden_states
        vis_dim = self.model_vision.embed_dim
        text_dim = config_opt.hidden_size
        self.connector = nn.ModuleList([nn.Linear(vis_dim, text_dim) for i in range(injected_hidden_states)])
        
        
    def forward(self, image=None, text=None, mode='generate', return_dict=True, labels=None, reduction='mean', **generation_kwargs):
        
        if image is not None:
            image_embed, image_feat = self.model_vision(image, external_features=None)

            image_feat = list(image_feat)
            image_feat = image_feat[-self.injected_hidden_states:]

            ## only cls token,  we can think of somthing else
            for i in range(1, self.injected_hidden_states + 1):
                image_feat[-i] = self.connector[-i](image_feat[-i][:, 0, :].unsqueeze(1))
        else:
            image_feat = None

            
        # image_feat = None
        if mode == 'train' or mode == 'evaluate':
            text_output = self.model_text(input_ids=text.input_ids, attention_mask=text.attention_mask, return_dict=return_dict, vis_prefix=image_feat, labels = labels, reduction=reduction)
            return text_output
        elif mode == 'generate':
            print('generation')
            gen = self.model_text.generate(input_ids=text.input_ids, vis_prefix=image_feat, **generation_kwargs)        
            return gen
            