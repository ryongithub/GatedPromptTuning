#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_moco import VisionTransformerMoCo
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class PromptedVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        
        # define temperature for attention shaping
        self.temp = self.prompt_config.TEMP
        self.temp_learn = self.prompt_config.TEMP_LEARN
        if self.temp_learn:
            self.temp = nn.Parameter(torch.ones(prompt_config.TEMP_NUM))

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # define block-wise learnable gate scalar
        if self.prompt_config.GATE_PRIOR:       
            gate_logit = (-torch.ones(self.prompt_config.GATE_NUM) * self.prompt_config.GATE_INIT)        
            self.gate_logit = nn.Parameter(gate_logit)
            print(self.gate_logit)
       
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((
                cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
                
    def reinit_temp(self):
        assert self.temp_learn, "reinit_temp() could be run only when config.TEMP_LEARN == True"
        self.temp.data.copy_(self.temp.data.clamp(min=self.prompt_config.TEMP_MIN, max=self.prompt_config.TEMP_MAX))

    def forward_features(self, x):
        x = self.incorporate_prompt(x)
            
        # deep
        if self.prompt_config.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    x = torch.cat((
                        x[:, :1, :],
                        self.prompt_dropout(
                            self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                        ),
                        x[:, (1 + self.num_tokens):, :]
                    ), dim=1)
                    x = self.blocks[i](x)
               
        else:
            # clamp temperatures not to be too small or too large
            if self.temp_learn:
                self.reinit_temp()

            for i, blk in enumerate(self.blocks):
                # current block's input prompt representation
                if self.prompt_config.GATE_PRIOR and i < self.gate_logit.shape[0]:
                    gate = self.gate_logit[i].sigmoid()
                    prompt_in = x[:, 1: 1+self.prompt_config.NUM_TOKENS, :]

                # block-wise learnable temperature
                temp = self.temp if not isinstance(self.temp, nn.Parameter) else self.temp[i]
                
                x = blk(x, temp=temp)
                if self.prompt_config.GATE_PRIOR and i < self.gate_logit.shape[0]:
                    # current block's output prompt representation
                    prompt_out = x[:, 1: 1+self.prompt_config.NUM_TOKENS, :]
                    # convex combinate input and output prompt representations of current block via learnalbe gate
                    x = torch.cat([
                        x[:, 0:1, :], 
                        gate * prompt_out + (1 - gate) * prompt_in, 
                        x[:, 1+self.prompt_config.NUM_TOKENS:, :]
                    ], dim=1)

        norm_func = self.norm 
        if self.prompt_config.VIT_POOL_TYPE == "imgprompt_pool":
            assert self.prompt_config.LOCATION == "prepend"
            outcome = norm_func(x[:, 1:, :].mean(dim=1))  # global pool without cls token
            
        elif self.prompt_config.VIT_POOL_TYPE == "original":
            x = norm_func(x)
            outcome = x[:, 0]
            
        elif self.prompt_config.VIT_POOL_TYPE == "img_pool":
            assert self.prompt_config.LOCATION == "prepend"
            outcome = norm_func(x[:, self.num_tokens+1:, :].mean(dim=1))
        elif self.prompt_config.VIT_POOL_TYPE == "prompt_pool":
            assert self.prompt_config.LOCATION == "prepend"
            outcome = norm_func(x[:, 1:self.num_tokens+1, :].mean(dim=1))
           
        else:
            raise ValueError("pooling type for output is not supported")
    
        return outcome


def build_model(model_type, prompt_cfg):
    if "vitb" in model_type:
        return vit_base(prompt_cfg)
    elif "vits" in model_type:
        return vit_small(prompt_cfg)
    

def vit_small(prompt_cfg, **kwargs):
    model = PromptedVisionTransformerMoCo(
        prompt_cfg,
        patch_size=16, embed_dim=384, depth=12, drop_path_rate=0.1,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(prompt_cfg, **kwargs):
    model = PromptedVisionTransformerMoCo(
        prompt_cfg,
        patch_size=16, embed_dim=768, depth=12, drop_path_rate=0.1,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

