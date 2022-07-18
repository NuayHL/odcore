import torch
import numpy as np
from utils.ema import ModelEMA

class Train():
    def __init__(self, config, args, model, rank):
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

    def train(self):
        pass

    def load_ckpt(self):
        if self.args.ckpt_file is '': pass
        else:
            self.model =
    def save_ckpt(self):
        pass
    def init_logger(self):
        pass
    def build_dataloader(self):
        pass
    def load_checkpoint(self):
        pass
    def build_optimizer(self):
        pass
    def using_DDP(self, args, model):
        pass
if __name__ == "__main__":
    pass