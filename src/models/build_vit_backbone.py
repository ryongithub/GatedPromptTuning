#!/usr/bin/env python3
import numpy as np
import torch
import os
from .vit_backbones.vit_moco import vit_base as moco_vit_model
from .vit_backbones.vit_mae import build_model as mae_vit_model

from .vit_prompt.vit_moco import build_model as prompt_moco_vit
from .vit_prompt.vit_mae import build_model as prompt_mae_vit


MODEL_ZOO = {
    "mae_vitb16": "mae-ViT-B.pth",
    "mae_vitl16": "mae-ViT-L.pth",
    "mocov3_vitb16" : "mocov3-ViT-B.pth.tar",
    "mocov3_vits16" : "mocov3-ViT-S.pth.tar",
}



def build_mae_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
):  
    if not model_type in ["mae_vitb16", "mae_vitl16"]:
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_mae_vit(model_type, prompt_cfg)
    else:
        model = mae_vit_model(model_type)
    out_dim = model.embed_dim

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_mocov3_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
):
    if not model_type in ["mocov3_vitb16", "mocov3_vits16"]:
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_moco_vit(model_type, prompt_cfg)
    else:
        model = moco_vit_model()
        
    out_dim = 384 if model_type.endswith('s16') else 784
    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            key = k.replace('module.', '')
            if key.startswith('base_encoder.'):
                key = key.replace('base_encoder.', '')
            elif key.startswith('momentum'):
                del state_dict[k]
                continue
            state_dict[key] = state_dict[k]
    
        # delete renamed or unused k
        del state_dict[k]
   
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.head = torch.nn.Identity()
    return model, out_dim

