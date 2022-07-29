import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils.ema import ModelEMA
from data.dataloader import build_dataloader

class Train():
    def __init__(self, config, args, model: nn.Module, rank):
        self.args = args
        self.config = config
        self.model = model
        self.rank = rank
        self.is_main_process = True if rank in [-1,0] else False
        if rank == -1:
            self.device = 'cuda'
        else:
            self.device = rank
        self.ema = ModelEMA(model) if self.is_main_process else None
        self.train_img_path = config.training.train_img_path
        self.train_img_anns_path = config.training.train_img_anns_path
        self.val_img_path = config.training.val_img_path
        self.val_img_anns_path = config.training.val_img_anns_path

    def go(self):
        self.optimizer = self.build_optimizer()
        self.schedular

    def train(self):
        self.model.train()
        

    def load_ckpt(self):
        if self.args.ckpt_file is '': pass
        else:
            self.model = self.
    def save_ckpt(self):
        pass
    def init_logger(self):
        pass
    def build_dataloader(self):
        pass
    def load_checkpoint(self):
        pass
    def build_optimizer(self):
        if self.config.training.optimizer.type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr = self.config.training.optimizer.lr)
        elif self.config.training.optimizer.type.lower() == 'adam':
            optimizer = optim.AdamW(self.model.parameters(),)
        return optimizer
    def build_schedular(self):
        if not hasattr(self, 'optimizer'):
            self.optimizer = self.build_optimizer()
        schedular =
        return schedular

    def using_DDP(self, args, model):
        pass
if __name__ == "__main__":
    pass