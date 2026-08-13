import os
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import importlib
import cv2
from PIL import Image
import subprocess
import math

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if config["target"] == "base._pipeline.StableDiffusionXLPipeline":
        return get_obj_from_str(config["target"]).from_pretrained(**config.get("params", dict()) if config.get("params", dict()) else {})
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()) if config.get("params", dict()) else {})

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key):
            if getattr(args, key) != None:
                config[key] = getattr(args, key)
    for key in args.__dict__.keys():
        config[key]=getattr(args, key)
    return config


def get_device(gpu_ids):
    # Under a scheduler (LSF) CUDA_VISIBLE_DEVICES already restricts this process to
    # its allocated GPUs and torch renumbers them from 0, but nvidia-smi still reports
    # physical indices -- so 'auto' can return an index torch cannot see. UBP_GPU
    # overrides the choice; otherwise trust the scheduler's allocation.
    override = os.environ.get('UBP_GPU')
    if override:
        gpu_ids = override
    if gpu_ids == 'auto' and os.environ.get('CUDA_VISIBLE_DEVICES'):
        return 0
    if gpu_ids=='auto':
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu', '--format=csv,noheader,nounits'])
        gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
        gpu_info = []
        for line in gpu_info_lines:
            gpu_data = line.strip().split(', ')
            index, memory_free, temperature = map(int, gpu_data)
            gpu_info.append((index, memory_free, temperature))
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        
        memeory_rank_num=math.ceil(0.4*len(gpu_info))
        selected_gpus = gpu_info[:memeory_rank_num]
        selected_gpus.sort(key=lambda x: x[2])
        selected_device = selected_gpus[0][0]
        # device = torch.device(f'cuda:{selected_device}')
    elif gpu_ids=="cpu":
        device = torch.device('cpu')
    else:
        gpu_ids = list(map(int,gpu_ids.split(",")))
        selected_device=gpu_ids[0]
        # device = torch.device(f'cuda:{selected_device}')
    return selected_device

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
       
    def compute_ranking_weights(self,loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1 / (i + 1)
        return weights
    
    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')

        # total_loss = (image_loss + text_loss) / 2
        
        return image_loss,text_loss, logits_per_image
