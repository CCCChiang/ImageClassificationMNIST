# data preprocess
import math
import random
import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset

# image
from PIL import Image, ImageOps, ImageEnhance
import numbers

# plot
import matplotlib.pyplot as plt

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)
    