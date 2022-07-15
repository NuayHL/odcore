import torch
import numpy as np

class Train():
    def __init__(self, cfg_train, model, rank):
        self.model = model
        self.rank = rank
        if rank == -1:
            self.device = 'cuda'
        else:
            self.device = rank
        self.train_img_path = cfg_train.train_img_path
        self.train_img_anns_path = cfg_train.train_img_anns_path
        self.val_img_path = cfg_train.val_img_path
        self.val_img_anns_path = cfg_train.val_img_anns_path

    def train(self):
        pass

    def load_ckpt(self):
        pass
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

if __name__ == "__main__":
    pass