import math
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import yaml
import torch.optim as optim

import config
from config import CN
from data.dataloader import build_dataloader
from utils.ema import ModelEMA
from utils.exp_storage import mylogger
from utils.misc import progressbar

class Train():
    main_log_storage_path = 'running_log'
    def __init__(self, config: CN, args, model: nn.Module, rank):
        self.args = args
        self.config = config
        self.rank = rank
        self.model = model
        self.pre_exp_setting()

    def pre_exp_setting(self):
        self.is_main_process = True if self.rank in [-1,0] else False
        self.set_log_path()
        self.set_logger()
        if self.args.ckpt_file != '' and self.args.fine_tune != '':
            print("Warning: Detect both ckpt_file and fine_tune file, using ckpt_file")
            self.args.fine_tune = ''
        self.dump_configurations()
        if self.rank == -1:
            self.device = 'cuda'
        else:
            self.device = self.rank

    def set_log_path(self):
        if self.is_main_process:
            base_name = self.config.exp_name
            print('Experiment Name: %s'%base_name)
            if not os.path.exists(self.main_log_storage_path):
                os.mkdir(self.main_log_storage_path)
            name_index = 1
            final_name = base_name
            while (final_name in os.listdir(self.main_log_storage_path)):
                final_name = base_name+'_'+str(name_index)
                name_index += 1
            self.exp_log_name = final_name
            final_name = self.main_log_storage_path+'/'+final_name
            os.mkdir(final_name)
            self.exp_log_path = final_name
            print('Experiment Storage Path: %s'%self.exp_log_path)

    def set_logger(self):
        if self.is_main_process:
            self.logger = mylogger(self.exp_log_name, self.exp_log_path)

    def dump_configurations(self):
        self.config.dump_to_file(self.exp_log_name+"_cfg", self.exp_log_path)
        self.args_to_file(self.exp_log_name+'_args', self.exp_log_path)

    def args_to_file(self, yaml_name='args', path=''):
        arg_dict = vars(self.args)
        with open(os.path.join(path, yaml_name+'.yaml'),'w') as f:
            yaml.dump(arg_dict, f)

    def go(self):
        self.load_finetune_model()
        self.build_optimizer()
        self.build_scheduler()
        self.ema = ModelEMA(self.model) if self.is_main_process else None
        self.load_ckpt()
        self.load_model_to_GPU()
        self.final_epoch = self.config.training.final_epoch
        assert self.final_epoch > self.start_epoch
        self.batchsize = self.args.batch_size
        self.build_train_dataloader()
        self.build_val_dataloader()
        print('==========go==========')
        self.train()

    def train(self):
        itr_in_epoch = len(self.train_loader)
        for epoch in range(self.start_epoch, self.final_epoch + 1):
            self.model.train()
            if self.rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            for i, samples in enumerate(self.train_loader):
                samples['imgs'] = samples['imgs'].to(self.device).float() / 255
                samples['annss'] = samples['annss'].to(self.device)
                loss, loss_dict = self.model(samples)
                progressbar(i/float(itr_in_epoch), endstr=)


    def val(self):
        self.model.eval()
        
    def load_finetune_model(self):
        print('FineTuning Model:%', end='')
        if self.args.fine_tune != ' ':
            print(self.args.fine_tune)
            print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.fine_tune)
                self.model.load_state_dict(ckpt_file['model'])
                print('SUCCESS')
                self.logger.info('Using FineTuning Model: %s'%self.args.fine_tune)
            except:
                print("FAIL")
                raise
        else:
            print('None')

    def load_ckpt(self):
        print('Checkpoint File:', end=' ')
        if self.args.ckpt_file != '':
            print(self.args.ckpt_file)
            print("\t-Loading:", end=' ')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                self.model.load_state_dict(ckpt_file['model'])
                self.start_epoch = ckpt_file['last_epoch'] + 1
                self.optimizer.load_state_dict(ckpt_file['optimizer'])
                self.scheduler.load_state_dict(ckpt_file['schedular'])
                if self.is_main_process:
                    self.ema.ema = ckpt_file['ema']
                    self.ema.updates = ckpt_file['updates']
                print('SUCCESS')
                self.logger.info('Using Checkpoint: %s'%self.args.ckpt_file)
            except:
                print("FAIL")
                raise
        if self.args.ckpt_file == '':
            print("None")
            self.start_epoch = 1
        print("Start epoch:", self.start_epoch)
        self.logger.info("Start Epoch: %d"%self.start_epoch)

    def save_ckpt(self, file_name):
        if self.is_main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'],)
            ckpt = {}
            ckpt['last_epoch'] = self.current_epoch
            ckpt['model'] = self.model.state_dict()
            ckpt['optimizer'] = self.optimizer.state_dict()
            ckpt['schedular'] = self.scheduler.state_dict()
            ckpt['ema'] = self.ema.ema
            ckpt['ema_updates'] = self.ema.updates
            torch.save(ckpt, self.exp_log_path+'/'+file_name+'.pth')

    def load_model_to_GPU(self):
        if self.rank == -1:
            self.model = self.model.to(self.device)
            print("Using Single GPU")
        else:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
            print("DDP mode, Using multiGPU")

    def build_train_dataloader(self):
        self.train_loader = build_dataloader(self.config.training.train_img_anns_path,
                                             self.config.training.train_img_path,
                                             self.config.data,
                                             self.batchsize, self.rank, self.args.workers,
                                             'train')

    def build_val_dataloader(self):
        if self.is_main_process:
            self.val_laoder = build_dataloader(self.config.training.val_img_anns_path,
                                               self.config.training.val_img_path,
                                               self.config.data,
                                               self.batchsize, -1, self.args.workers,
                                               'val')

    def build_optimizer(self):
        config_opt = self.config.training.optimizer

        g_bnw, g_w, g_b = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g_b.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g_bnw.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g_w.append(v.weight)

        if self.config.training.optimizer.type.lower() == 'sgd':
            self.optimizer = optim.SGD(g_bnw, lr = config_opt.lr, momentum=config_opt.momentum, nesterov=True)
        elif self.config.training.optimizer.type.lower() == 'adam':
            self.optimizer = optim.Adam(g_bnw, lr = config_opt.lr, betas=(config_opt.momentum, 0.999))
        else:
            raise NotImplementedError

        self.optimizer.add_param_group({'params': g_w, 'weight_decay': config_opt.weight_decay})
        self.optimizer.add_param_group({'params': g_b})

    def build_scheduler(self):
        if not hasattr(self, 'optimizer'):
            self.build_optimizer()
        if self.config.training.schedular == 'cosine':
            lf = lambda x: ((1 - math.cos(x * math.pi / self.final_epoch)) / 2) * (self.config.training.schedular.lrf - 1) + 1
        else:
            raise NotImplementedError
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

if __name__ == "__main__":
    pass