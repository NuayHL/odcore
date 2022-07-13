import torch
import numpy as np

class Train():
    def __init__(self, cfg, rank):
        self.rank = rank
        if rank == -1:
            self.device = 'cuda'
        else:
            self.device = rank
        self.train_img_path = cfg.train_img_path
        self.train_img_anns_path = cfg.train_img_anns_path
        self.val_img_path = cfg.val_img_path
        self.val_img_anns_path = cfg.val_img_anns_path

    def train(self):
        pass

    def build_dataloader(self):
        pass
    def load_checkpoint(self):
        pass
    def build_model(self):
        pass
    def build_optimizer(self):
        pass

if __name__ == "__main__":
    pass