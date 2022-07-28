import torch
import torch.nn as nn
import math
import numpy as np
from copy import deepcopy

def seed_init(num=None):
    if num == None: return 0
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)

def x1y1wh_x1y1x2y2_(labels):
    '''labels: np.ndarray or torch.Tensor with shape:n x (4+)'''
    labels[:,2] += labels[:,0]
    labels[:,3] += labels[:,1]

def x1y1x2y2_x1y1wh_(labels):
    '''labels: np.ndarray or torch.Tensor with shape:n x (4+)'''
    labels[:,2] -= labels[:,0]
    labels[:,3] -= labels[:,1]

def x1y1wh_x1y1x2y2(labels):
    '''labels: np.ndarray or torch.Tensor with shape:n x (4+)'''
    new_labels = deepcopy(labels)
    new_labels[:,2] += new_labels[:,0]
    new_labels[:,3] += new_labels[:,1]
    return new_labels

def x1y1x2y2_x1y1wh(labels):
    '''labels: np.ndarray or torch.Tensor with shape:n x (4+)'''
    new_labels = deepcopy(labels)
    new_labels[:,2] -= new_labels[:,0]
    new_labels[:,3] -= new_labels[:,1]
    return new_labels
