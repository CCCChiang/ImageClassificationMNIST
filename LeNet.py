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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 16*4*4, out_features = 120), 
            nn.Sigmoid(),
            nn.Linear(in_features = 120, out_features = 84),
            nn.Sigmoid(),
            nn.Linear(in_features = 84, out_features = 10)
        )
    def forward(self, img):
        feature = self.features(img)
        output = self.classifier(feature.view((feature.shape[0], -1)))
        return output