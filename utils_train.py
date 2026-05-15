import sys
import argparse
from typing import Any
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pickle
import io
import pdb






























class cifar10_training_data(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data, label = self.data[idx]

        if isinstance(img_data, bytes):
            img = Image.frombytes('RGB', (32, 32), img_data)
            img = img.convert('RGB')

        else:
            if torch.is_tensor(img_data):
                img_data = img_data.cpu().numpy()

            img_data = np.array(img_data)
            img_data = img_data.transpose(1, 2, 0) * 255
            img_data = img_data.astype(np.uint8)
            img = Image.fromarray(img_data.clip(0, 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label)






def add_random_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    noisy_tensor = torch.clamp(tensor + noise, 0, 1)
    return noisy_tensor

def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace





