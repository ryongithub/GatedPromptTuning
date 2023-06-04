# Gated Prompt Tuning
This is the official PyTorch implementation for "Improving Visual Prompt Tuning for Self-supervised Vision Transformers" [ICML 2023].

This repository is heavily based on the official PyTorch implementation of "Visual Prompt Tuning" [ECCV 2022] : [KMnp/vpt](https://github.com/KMnP/vpt).

<!-- Gated Prompt Tuning proposes an improved prompt tuning method for self-supervised Vision Transformers via introduced learnable scalar gate for each ViT block. With the learnable gates, the prompt tokens selectively interact with task-relevant blocks so that it achieves better transfer performances for self-supervised Vision Transformers.  -->

# Requirements
- python 3.8.12
- PyTorch 1.7.1
- torchvision 0.8.2
- timm 0.5.4
- CUDA 11.0
- RTX 8000 GPU

# Environment setup
```
conda create -n [ENV_NAME] python=3.8.12 -y
conda activate [ENV_NAME]
bash env_install.sh
```

# Data preparation
- FGVC : The datasets should be located in the 'data' folder (CUB, OxfordFlowers, StanfordCars, StanfordDogs, NABirds) 
- VTAB : Please refer to [`VTAB_SETUP.md`] (in accordance with [KMnp/vpt](https://github.com/KMnP/vpt))
- A more detailed guideline for data preparation will be updated soon.

# Pretraiend SSL ViTs
- pretrained checkpoints for MAE, MoCo-v3 should be located in the 'params' folder.

 # Run experiments
 ```
 bash run.sh [data_name] [encoder] [batch_size] [base_lr] [num_tokens] [gate_init]
 ```
 For example for the CUB dataset, execute
 ```
 bash run.sh cub mae_vitb16 64 0.1 100 5
 ```