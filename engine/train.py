import torch
import torch.nn as nn
import numpy as np
import os.path as path
import torch.optim as optim
from utils.ema import ModelEMA
from data.dataloader import build_dataloader
from utils.exp_storage import mylogger

class Train():
    def __init__(self, config, args, model: nn.Module, rank):
        self.args = args
        self.config = config
        self.rank = rank
        self.is_main_process = True if rank in [-1,0] else False
        self.logger = mylogger() if self.is_main_process else None
        if rank == -1:
            self.device = 'cuda'
        else:
            self.device = rank
        self.model = model

    def go(self):
        self.load_model()
        self.optimizer = self.build_optimizer()
        self.schedular = self.build_schedular()
        self.ema = ModelEMA(self.model) if self.is_main_process else None
        self.load_ckpt()
        self.current_epoch =
        self.final_epoch =
        self.batchsize =

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

    def val(self):
        self.model.eval()
        
    def load_model(self):
        if self.args.fine_tune != '':
            print('FineTuning Model:%', end='')
            try:
                self.model.load_state_dict()
            except:
                print("")
    def load_ckpt(self):
        print('Checkpoint File:', end='')
        key_load_flag = True
        other_load_flag = True
        if self.args.ckpt_file != '':
            print(self.args.ckpt_file)
            print("Start Loading...")

            print("\t-ckpt file:", end='')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                print('SUCCESS')
            except:
                print("FAIL")
                key_load_flag = False
            print("\t-model dict:", end='')
            try:
                self.model.load_state_dict(ckpt_file['model'])
                print("SUCCESS")
            except:
                print("FAIL")
                key_load_flag = False
            print("\t-starting epoch:", end='')
            try:
                self.start_epoch = ckpt_file['current_epoch']
                print(self.start_epoch)
            except:
                print("FAIL")
            if self.is_main_process:
                print("\t-ema:", end='')
                try:
                    self.ema.ema = ckpt_file['ema']
                    self.ema.updates = ckpt_file['updates']
                    print("SUCCESS")
                except:
                    print("FAIL")
                print("\t-optimizer:", end='')
        if not key_load_flag:
            print("FATAL ERROR DETECT\nENDING TRAINING PROCESS")
            exit()
        if self.args.ckpt_file == '':
            print("None")
            self.start_epoch = 1


        print("Start epoch:", self.start_epoch)

    def save_ckpt(self):
        if self.is_main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'],)
            ckpt = {}
            ckpt['epoch'] = self.current_epoch
            ckpt['model'] = self.model.state_dict()
            ckpt['optimizer'] = self.optimizer.state_dict()
            ckpt['ema'] = self.ema.ema
            ckpt['updates'] = self.ema.updates
            torch.save(ckpt, self.exp_path)

    def init_logger(self):
        pass
    def build_train_dataloader(self):
        self.train_loader = build_dataloader()
    def build_val_dataloader(self):
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