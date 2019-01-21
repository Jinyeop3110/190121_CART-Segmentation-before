import numpy
import scipy
from scipy import io
import random
import math
import os
import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from visdom import Visdom


class Model(nn.Module):
    def __init__(self, name, input_dim, output_dim, hidden_dims=[32, 32], use_batchnorm=True, dropout_p=0,
                 activation_fn=nn.ReLU):
        super(Model, self).__init__()
        self.layer_dims = [input_dim] + hidden_dims
        self.activation_fn = activation_fn
        self.layers = nn.Sequential()
        for i, h_dim in enumerate(self.layer_dims[0:len(self.layer_dims) - 2]):
            layer_temp = nn.Sequential(
                nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]),
                nn.BatchNorm1d(self.layer_dims[i + 1]),
                self.activation_fn()
            )
            self.layers = nn.Sequential(
                self.layers,
                layer_temp
            )

        self.layer_last = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.layer_dims[len(self.layer_dims) - 1], output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = self.layer_last(out);
        return out