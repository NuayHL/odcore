import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from data.dataset import CocoDataset
from utils.visualization import *
from utils.misc import *
from engine.train import Train
from config import get_default_cfg
from args import get_train_args_parser
from data.data_augment import *
cfg = get_default_cfg()


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv2(self.bn(self.conv1(x))))

# model = DummyModel()
# m = model.modules()
# for ms in m:
#     if "Conv" in ms.__class__.__name__:
#         print(ms.bias)
#         nn.init.uniform_(ms.bias, 1.0,1.0)
#         print(ms.bias)
