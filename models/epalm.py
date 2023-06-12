
import torch
from torch import nn

from transformers import AutoConfig
from models.opt import OPTForCausalLM
import models.vit 

import numpy as np

from copy import deepcopy


import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding

from models.connector import connector 

from models.adapters import (
    Adapter,
    ParallelAdapter,
    AdapterWrapper,
    ParallelAdapterWrapper,
)
from typing import Literal



from models.timesformer import TimeSformer


from models.ast import ASTModel  


def rank_answer(model, image, question_input, answer_ids, answer_atts, k, tokenizer, special_answer_token=None):

    num_ques = question_input.input_ids.size(0)
    if special_answer_token is not None:
        start_input = question_input
        start_ids = question_input.input_ids
        attention_mask = question_input.attention_mask
    else:
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
    
    attention_mask = tile(attention_mask, 0, k)
    image = tile(image, 0, k)
    
    
        
    start_ids = tile(start_ids, 0, k)
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



        
     


## modified from https://github.com/ylsung/VL_adapter/blob/main/VL-T5/src/prompt/prompt_modeling.py

class InputPrompts(nn.Module):
    def __init__(self, prompt_len = 10,
                 prompt_dim = 1024,
                 mid_dim=512, mlp=True, deep=False, nb_prompts=12):
        super().__init__()
        
        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.mid_dim = mid_dim

        

        self.deep = deep 
        self.nb_prompts = nb_prompts
        if self.deep:
            print("Init deep prompts", nb_prompts)
            p_len = prompt_len*nb_prompts
        else:
            p_len = prompt_len

        self.prefix_tokens = torch.arange(p_len).long()
        if mlp:
            self.prefix_embedding = nn.Sequential(
                nn.Embedding(p_len, self.prompt_dim),
                nn.Linear(self.prompt_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.prompt_dim),
            )
        else:
            self.prefix_embedding = nn.Sequential(
                nn.Embedding(p_len, self.prompt_dim),
            )

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, pdim)
        
        if self.deep:

            prefix_prompt = prefix_prompt.view(bsz, self.nb_prompts, self.prompt_len, self.prompt_dim)
            prompts = [prefix_prompt[:, i, :, :] for i in range(self.nb_prompts)]
            return prompts

        return prefix_prompt


class ePALM(nn.Module):
    def __init__(self,                 
                 opt_model_name = 'facebook/opt-350m',
                 vision_model_name = 'vit_base_patch16_224',
                 use_vis_prefix = True,
                 start_layer_idx = 11,
                 end_layer_idx = 23,
                 return_hidden_state_vision = True,
                 config = None, low_cpu=False,
                 ):
        super().__init__()
        print("Loading ePALM ...")
        # text

        config_opt = AutoConfig.from_pretrained(opt_model_name)
        
        config_opt.use_vis_prefix = use_vis_prefix
        config_opt.start_layer_idx = start_layer_idx
        config_opt.end_layer_idx = end_layer_idx
            
        use_cache = config.get('use_cache', True)
        config_opt.use_cache = use_cache 




        text_step = config.get('text_step', 1)
        config_opt.text_step = text_step

        self.select_higher_step = config.get('select_higher_step', False)
        config_opt.select_higher_step = self.select_higher_step
        

        if not hasattr(config_opt, 'activation_dropout'):
            config_opt.activation_dropout = 0.0

        print("Loading: ", opt_model_name)
        self.no_attention_mask = False

        if low_cpu:
            self.model_text = OPTForCausalLM.from_pretrained(opt_model_name, config=config_opt, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=False)
        else:
            self.model_text = OPTForCausalLM.from_pretrained(opt_model_name, config=config_opt)
            
        self.transformer = self.model_text.model.decoder.layers

        print(self.model_text.config)
        # vision
        print("Loading: ", vision_model_name)

        image_size = config.get('image_res', 224)
        num_frames = config.get('num_frames', 4)
        pretrained_model = config.get('pretrained_model', None)


        mask_p = config.get('mask_p', 0)
        
        space_only_for_images = config.get('space_only_for_images', None)
        if 'timesformer' in vision_model_name:
            print("Load:", pretrained_model)
            self.model_vision = TimeSformer(img_size=image_size, num_frames=num_frames, 
            attention_type='divided_space_time',  pretrained_model=pretrained_model, 
            return_hidden_state=return_hidden_state_vision, space_only_for_images=space_only_for_images)
            vis_dim = self.model_vision.embed_dim


        elif 'ast' in vision_model_name:
            print("Load:", pretrained_model)
            self.model_vision = ASTModel(audioset_pretrain=True, verbose=True, 
                pretrained_model=pretrained_model, return_hidden_state=return_hidden_state_vision)
            vis_dim = self.model_vision.original_embedding_dim

        else: 
            vision_func = getattr(models.vit, vision_model_name)
            if pretrained_model is not None:
                pretrained=False
            else:
                pretrained = True
            self.model_vision = vision_func(pretrained=pretrained, return_hidden_state=return_hidden_state_vision, 
                 mask_p=mask_p)
            if pretrained_model:
                self.model_vision.load_pretrained(pretrained_model)

            vis_dim = self.model_vision.embed_dim
        
        # connector
        connector_type = config.get('connector_type', 'linear')
        self.connector_type = connector_type





        injected_hidden_states = config.get('injected_hidden_states', 1)
        self.injected_hidden_states = injected_hidden_states
        
        
        text_dim = self.model_text.config.hidden_size

        connector_config = config.get('connector_config', None)
        self.shared_connector = config.get('shared_connector', None)


            
        if self.shared_connector is not None:
            num_connectors = 1 
        else:
            num_connectors = self.injected_hidden_states


        self.connector = connector(connector_type=connector_type, input_dim=vis_dim, output_dim=text_dim, num_layers=num_connectors, connector_config=connector_config) #nn.ModuleList([nn.Linear(vis_dim, text_dim) for i in range(injected_hidden_states)])

        # Prompt
        self.prompt_tuning = config.get('prompt_tuning', False)
        if self.prompt_tuning:
            prompt_len = config.get("prompt_len", 10)

            prompt_dim = config_opt.word_embed_proj_dim

            mlp = config.get('mlp', True)
            deep = config.get('deep', False)
            nb_prompts = config.get('nb_prompts', 12)
            self.prompt_module = InputPrompts(prompt_len=prompt_len, prompt_dim=prompt_dim, mid_dim=prompt_dim, 
                mlp=mlp, deep=deep, nb_prompts=nb_prompts)

        # Adapters
        self.use_adapters = config.get('use_adapters', False)
        self.mlp_adapter_added, self.attn_adapter_added = False, False
        if self.use_adapters:
            mlpconfig = config['adapter_config'].get("mlp", None)
            if mlpconfig is not None:
                mlp_config = deepcopy(mlpconfig)
            else:
                mlp_config = mlpconfig

            ff_attr = "fc2"
            attn_attr = "self_attn"

            if mlp_config:
                assert mlp_config.get("adapter_type") is not None
                self.add_adapters(
                    location="mlp",
                    adapter_type=mlp_config.pop("adapter_type"),
                    downsample_factor=mlp_config.pop("downsample_factor", 4),
                    ff_attr = ff_attr,
                    attn_attr = attn_attr,
                    **mlp_config,
                )
            attn_config = deepcopy(config['adapter_config'].get("attention", None))
            if attn_config:
                assert attn_config.get("adapter_type") is not None
                self.add_adapters(
                    location="attention",
                    adapter_type=attn_config.pop("adapter_type"),
                    ff_attr = ff_attr,
                    attn_attr = attn_attr,
                    **attn_config,
                )
        
        
    def forward(self, image=None, text=None, mode='generate', return_dict=True, labels=None, reduction='mean', modality=None, **generation_kwargs):
        
        if image is not None:
            image_embed, image_feat = self.model_vision(image, external_features=None)

            image_feat = list(image_feat)

            image_feat = image_feat[-self.injected_hidden_states:]
            
            for i in range(1, self.injected_hidden_states + 1):

                if self.shared_connector:
                    image_feat[-i] = self.connector[0](image_feat[-i][:, 0, :].unsqueeze(1))
                else:
                    if modality is not None:
                        image_feat[-i] = self.connector[-i](image_feat[-i][:, 0, :].unsqueeze(1), modality=modality)
                    else:
                        image_feat[-i] = self.connector[-i](image_feat[-i][:, 0, :].unsqueeze(1))

        else:
            image_feat = None
        
        if self.prompt_tuning:
            prompts = self.prompt_module.get_prompt(text.input_ids.shape[0], text.attention_mask.device)
        else:
            prompts = None 

        if self.no_attention_mask:
            attention_mask = None 
        else:
            attention_mask = text.attention_mask
        if mode == 'train' or mode == 'evaluate':
            text_output = self.model_text(input_ids=text.input_ids, attention_mask=attention_mask, 
                return_dict=return_dict, vis_prefix=image_feat, labels = labels, reduction=reduction, 
                prompt_embeds=prompts, connector=self.connector)
            return text_output
        elif mode == 'generate':
            gen = self.model_text.generate(input_ids=text.input_ids, vis_prefix=image_feat, prompt_embeds=prompts, 
                connector=self.connector, attention_mask=attention_mask,
                **generation_kwargs)        
            return gen


    def add_adapters(
        self,
        downsample_factor: int = 4,
        adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "fc2",
        attn_attr: str = "self_attn",
        **adapter_kwargs,
    ):
        """
        Adds an adapter layer to `self` at the specified location
        """
        assert adapter_type in [
            "normal",
            "parallel",
            "scaled_parallel",
        ], "adapter_type must be one of 'normal', 'parallel', or 'scaled_parallel'"
        assert location in [
            "mlp",
            "attention",
        ], "location must be one of 'mlp' or 'attention'"

        for l in range(len(self.transformer)):
            if location == "mlp":
                if self.mlp_adapter_added:
                    raise ValueError("Adapter layer already added")
                mlp = getattr(self.transformer[l], ff_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapter(
                        module=mlp,
                        dim=self.model_text.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled=adapter_type == "scaled_parallel",
                        **adapter_kwargs,
                    )
                else:
                    adpt = Adapter(
                        dim=self.model_text.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                    adapter_layer = nn.Sequential(
                        *[
                            mlp,
                            adpt,
                        ]
                    )
                setattr(self.transformer[l], ff_attr, adapter_layer)
            else:
                if self.attn_adapter_added:
                    raise ValueError("Adapter layer already added")
                attn = getattr(self.transformer[l], attn_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapterWrapper(
                        module=attn,
                        dim=self.model_text.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled="scaled" in adapter_type,
                        **adapter_kwargs,
                    )
                else:
                    adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=self.model_text.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                setattr(self.transformer[l], attn_attr, adapter_layer)

        if location == "mlp":
            self.mlp_adapter_added = True
        else:
            self.attn_adapter_added = True
            

