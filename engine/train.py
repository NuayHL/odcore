import time
import math
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import yaml
import torch.optim as optim

from config import CN
from data.dataloader import build_dataloader
from data.data_augment import Normalizer
from utils.ema import ModelEMA
from utils.exp_storage import mylogger
from utils.misc import progressbar, loss_dict_to_str
from utils.exp import Exp
from engine.eval import coco_eval

# Set os.environ['CUDA_VISIBLE_DEVICES'] = '-1' and rank = -1 for cpu training
#
# model must have method model.set(args, device)

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
        self.check_and_set_device()
        self.using_resume = False if self.args.resume_exp == '' else True
        self.resume_from_file()
        self.set_log_path()
        if self.args.ckpt_file != '' and self.args.fine_tune != '':
            self.print("Warning: Detect both ckpt_file and fine_tune file, using ckpt_file")
            self.args.fine_tune = ''
        self.dump_configurations()
        self.set_logger()
        self.model.set(self.args, self.device)
        self.normalizer = Normalizer(self.config.data,self.device)

    def check_and_set_device(self):
        if self.is_main_process:
            try:
                available_device = os.getenv('CUDA_VISIBLE_DEVICES')
            except:
                raise
            available_device = [int(device) for device in available_device.strip().split(',')]
            if len(available_device) == 1:
                assert self.rank == -1,'Only find 1 Device, please set rank as -1 for training!'
                if available_device[0] == -1:
                    self.print('Warning: Using Device CPU')
                    self.device = 'cpu'
                    return
            self.print('Find Device:', available_device)
            self.print('Main Process runs at CUDA:%d'%available_device[0])
            self.device = 0
        else:
            self.device = self.rank

    def resume_from_file(self):
        if self.using_resume:
            self.print("Resume Experiment For Training")
            self.formal_exp = Exp(self.args.resume_exp,self.is_main_process)
            self.config.merge_from_file(self.formal_exp.get_cfg_path())
            self.args.ckpt_file = self.formal_exp.get_ckpt_file_path()

    def set_log_path(self):
        if self.is_main_process:
            if self.using_resume:
                self.exp_log_path = self.args.resume_exp
                self.exp_log_name = os.path.basename(self.args.resume_exp)
                self.exp_loss_log_name = self.exp_log_name + '_loss'
            else:
                base_name = self.config.exp_name
                self.print('Experiment Name: %s'%base_name)
                if not os.path.exists(self.main_log_storage_path):
                    os.mkdir(self.main_log_storage_path)
                name_index = 1
                final_name = base_name
                while (final_name in os.listdir(self.main_log_storage_path)):
                    final_name = base_name+'_'+str(name_index)
                    name_index += 1
                self.exp_log_name = final_name
                self.exp_loss_log_name = final_name + '_loss'
                final_name = os.path.join(self.main_log_storage_path, final_name)
                os.mkdir(final_name)
                self.exp_log_path = final_name
                self.print('Experiment Storage Path: %s'%self.exp_log_path)

    def set_logger(self):
        if self.is_main_process:
            if self.using_resume:
                if not self.formal_exp.log_file_name:
                    self.print('Making a new log file')
            self.logger = mylogger(self.exp_log_name, self.exp_log_path)
            if self.using_resume:
                if not self.formal_exp.log_loss_file_name:
                    self.print('Making a new loss log file for Resume Training')
                    with open(os.path.join(self.exp_log_path, self.exp_loss_log_name + '.log'), 'w') as fn:
                        print("Make new loss log for Resume Training")
                else:
                    if self.formal_exp.log_loss_file.incomplete_last_epoch:
                        self.print('Prepare for resume training, modifying loss log')
                        self.logger.info('Prepare for resume training, modifying loss log')
                        last_epoch = self.formal_exp.log_loss_file.last_epoch
                        self.formal_loss_log_name = self.exp_loss_log_name + '_e%d'%last_epoch
                        name_index = 1
                        while self.formal_loss_log_name+'.log' in self.formal_exp.get_exp_files():
                            self.formal_loss_log_name = self.exp_loss_log_name + '_' + str(name_index)
                            name_index += 1
                        self.formal_exp.log_loss_file.drop_incomplete_and_write(os.path.join(self.exp_log_path,
                                                                                             self.formal_loss_log_name+'.log'))
                        self.logger.info("Save the old loss log to %s"%self.formal_loss_log_name)
                        self.print("Save the old loss log to %s" % self.formal_loss_log_name)
                    del self.formal_loss_log_name
                del self.formal_exp
            self.logger_loss = mylogger(self.exp_loss_log_name, self.exp_log_path)

    def dump_configurations(self):
        if not self.using_resume and self.is_main_process:
            self.config.dump_to_file(self.exp_log_name+"_cfg", self.exp_log_path)
            self.args_to_file(self.exp_log_name+'_args', self.exp_log_path)

    def args_to_file(self, yaml_name='args', path=''):
        arg_dict = vars(self.args)
        with open(os.path.join(path, yaml_name+'.yaml'),'w') as f:
            yaml.dump(arg_dict, f)

    def pre_train_setting(self):
        self.final_epoch = self.config.training.final_epoch
        self.load_finetune_model()
        self.ema = ModelEMA(self.model) if self.is_main_process else None
        self.build_optimizer()
        self.build_scheduler()
        self.load_ckpt()
        assert self.final_epoch > self.start_epoch
        self.load_model_to_GPU()
        self.batchsize = self.config.training.batch_size
        self.print('Batch size:', self.batchsize)
        self.build_train_dataloader()
        self.val_setting()
        self.scaler = amp.GradScaler()
        self.itr_in_epoch = len(self.train_loader)

    def go(self):
        self.pre_train_setting()
        self.print('====================================== GO ======================================')
        self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.final_epoch + 1):
            self.current_epoch = epoch
            self.model.train()
            if self.rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            self.print('Epoch: %d/%d'%(self.current_epoch, self.final_epoch))
            time_epoch_start = time.time()
            for i, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                samples['imgs'] = samples['imgs'].to(self.device).float() / 255
                self.normalizer(samples)
                with amp.autocast(enabled=self.device != 'cpu'):
                    loss, loss_dict = self.model(samples)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema:
                    self.ema.update(self.model)
                loss_log = loss_dict_to_str(loss_dict)
                if self.is_main_process:
                    progressbar((i+1)/float(self.itr_in_epoch), barlenth=40, endstr=loss_log)
                    self.logger_loss.info('epoch '+str(self.current_epoch)+'/'+str(self.final_epoch)+
                                  ' '+loss_log)
            time_epoch_end = time.time()
            self.scheduler.step()
            if self.is_main_process:
                self.save_ckpt('last_epoch')
                self.logger.info('Complete epoch %d at %.1fmin, saving last_epoch.pth as last epoch'
                                 %(self.current_epoch,(time_epoch_end-time_epoch_start)/60))
            if self.current_epoch % self.config.training.eval_interval == 0:
                self.save_ckpt('epoch_%d'%self.current_epoch)
                if self.using_val:
                    self.valfun()
        self.save_ckpt('fin_epoch')
        self.log_info('Saving fin_epoch.pth')

    def val_coco(self):
        if self.val_loader == None or not self.is_main_process: return
        self.model.eval()
        if not hasattr(self, 'map'):
            self.map = 0.0
        if not hasattr(self, 'map50'):
            self.map50 = 0.0
        results = []
        self.print('Begin Evaluation')
        self.log_info('Evaluation begin at epoch %d'%self.current_epoch)
        time_start = time.time()
        for i, samples in enumerate(self.val_loader):
            samples['imgs'] = samples['imgs'].to(self.device).float() / 255
            self.normalizer(samples)
            with torch.no_grad():
                results.append(self.model(samples))

        time_end = time.time()
        self.print('mAP: %.2f, mAP50: %.2f'%(self.map, self.map50))
        self.log_info('Evaluation Complete at %.2f s, mAP: %.2f, mAP50: %.2f'
                      %(time_end-time_start, self.map, self.map50))

    def load_finetune_model(self):
        self.print('FineTuning Model: ', end='')
        if self.args.fine_tune != '':
            self.print(self.args.fine_tune)
            self.print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.fine_tune)
                self.model.load_state_dict(ckpt_file['model'])
                self.print('SUCCESS')
                self.log_info('Using FineTuning Model: %s'%self.args.fine_tune)
            except:
                self.print("FAIL")
                raise
        else:
            self.print('None')

    def load_ckpt(self):
        self.print('Checkpoint File:', end=' ')
        if self.args.ckpt_file != '':
            self.print(self.args.ckpt_file)
            self.print("\t-Loading:", end=' ')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                self.model.load_state_dict(ckpt_file['model'])
                self.start_epoch = ckpt_file['last_epoch'] + 1
                self.optimizer.load_state_dict(ckpt_file['optimizer'])
                self.scheduler.load_state_dict(ckpt_file['schedular'])
                if self.is_main_process:
                    self.ema.ema = ckpt_file['ema']
                    self.ema.updates = ckpt_file['ema_updates']
                self.print('SUCCESS')
                self.log_info('Using Checkpoint: %s'%self.args.ckpt_file)
            except:
                self.print("FAIL")
                raise
        if self.args.ckpt_file == '':
            self.print("None")
            self.start_epoch = 1
        self.print("Start epoch:", self.start_epoch)
        self.log_info("Start Epoch: %d"%self.start_epoch)

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

    def save_model(self, file_name):
        if self.is_main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'],)
            model_parameters = {}
            model_parameters['last_epoch'] = self.current_epoch
            model_parameters['model'] = self.model.state_dict()
            torch.save(model_parameters, self.exp_log_path+'/'+file_name+'.pth')

    def load_model_to_GPU(self):
        if self.rank == -1:
            self.model = self.model.to(self.device)
        else:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
            self.print("DDP mode, Using multiGPU")
        if self.ema:
            self.ema.ema = self.ema.ema.to(self.device)

    def build_train_dataloader(self):
        self.train_loader = build_dataloader(self.config.training.train_img_anns_path,
                                             self.config.training.train_img_path,
                                             self.config.data,
                                             self.batchsize, self.rank,
                                             self.config.training.workers,
                                             'train')

    def val_setting(self):
        if self.config.training.val_img_path == '':
            self.using_val = False
        else: self.using_val = True
        self.build_val_dataloader()

    def build_val_dataloader(self):
        if self.is_main_process:
            if self.using_val:
                self.val_type = self.config.training.val_metric
                if self.val_type == 'coco':
                    self.valfun = self.val_coco
                else:
                    raise NotImplementedError('Invalid Evaluation Metric')
                self.val_loader = build_dataloader(self.config.training.val_img_anns_path,
                                                   self.config.training.val_img_path,
                                                   self.config.data,
                                                   self.batchsize, -1, self.config.training.workers,
                                                   'val')
            else: self.val_loader = None

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
        if self.config.training.schedular.type == 'cosine':
            lf = lambda x: ((1 - math.cos(x * math.pi / self.final_epoch)) / 2) * (self.config.training.schedular.lrf - 1) + 1
        else:
            raise NotImplementedError
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def log_info(self,*args,**kwargs):
        if hasattr(self,'logger'):
            self.logger.info(*args,**kwargs)

if __name__ == "__main__":
    pass