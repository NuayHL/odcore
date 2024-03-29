import time
import json
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
from utils.paralle import de_parallel
from utils.optimizer import BuildOptimizer
from utils.lr_schedular import LFScheduler, LFScheduler_Step
from engine.eval import coco_eval, gen_eval

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
        self.check_and_set_train_type()
        self.set_log_path()
        self.dump_configurations()
        self.set_logger()

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
            self.print('Set Available Device:', available_device)
            self.print('Main Process runs at CUDA:%d'%available_device[0])
            self.device = 0
        else:
            self.device = self.rank
        self.using_DDP = False if self.rank == -1 else True

    def check_and_set_train_type(self):
        self.train_type = ''
        self.using_ckpt = True if self.args.ckpt_file != '' else False
        self.using_fine_tune = True if self.args.fine_tune != '' else False
        self.using_resume = True if self.args.resume_exp != '' else False
        self.check_resume()
        self.check_ckpt_finetune()
        self.print('Train Type: ', self.train_type)

    def check_resume(self):
        if self.using_resume:
            self.train_type = '[Resume]'
            self.formal_exp = Exp(self.args.resume_exp,self.is_main_process)
            self.config.merge_from_file(self.formal_exp.get_cfg_path())
            self.args.ckpt_file = self.formal_exp.get_ckpt_file_path()
            self.using_ckpt = True
            self.using_fine_tune = False

    def check_ckpt_finetune(self):
        if self.using_resume:
            return
        if self.using_ckpt:
            if self.using_fine_tune:
                self.print("Warning: Indicating both ckpt_file and fine_tune file, using ckpt_file")
                self.args.fine_tune = ''
                self.using_fine_tune = False
            self.train_type = '[Checkpoint]'
            return
        if self.using_fine_tune:
            self.train_type = '[FineTune]'
            return
        self.train_type = '[Scratch]'

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
                        self.print("Make new loss log for Resume Training")
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

    def go(self):
        self.pre_train_setting()
        self.print('====================================== GO ======================================')
        self.train()
        self.print('=================================== Complete! ==================================')
        self.summary()

    def pre_train_setting(self):
        self.log_info('Train Type: %s' % self.train_type)

        self.safety_mode = self.args.safety_mode
        self.using_loss_protect = self.safety_mode
        self.print('Safety Mode: ', end='')
        if self.safety_mode:
            self.print('[ON]')
        else:
            self.print('[OFF]')

        self.model.set(self.args, self.device)
        self.normalizer = Normalizer(self.config.data,self.device)

        self.batchsize = self.config.training.batch_size
        self.accumulate = int(self.config.training.accumulate)
        self.print('Batch size:%d, Accumulate:%d'%(self.batchsize, self.accumulate))
        self.print('\t-SimBatch size:%d'%(self.batchsize * self.accumulate))
        self.build_train_dataloader()
        self.itr_in_epoch = len(self.train_loader)

        self.final_epoch = self.config.training.final_epoch
        self.load_finetune_model()
        self.ema = ModelEMA(self.model) if self.is_main_process else None
        self.build_optimizer()
        self.build_scheduler()
        self.load_ckpt()
        assert self.final_epoch > self.start_epoch

        self.val_setting()
        self.load_model_to_device()
        self.load_optimizer_to_device()

        self.using_autocast = self.config.training.using_autocast and self.device != 'cpu'
        self.print('Using autocast:', self.using_autocast)
        self.using_warm_up = True if self.config.training.warm_up_steps != 0 else False
        self.print('Using warmup:', self.using_warm_up)
        self.warm_up_steps = max(500, self.config.training.warm_up_steps)
        if self.using_warm_up:
            self.print('\t-Warming step:', self.warm_up_steps)

        self.scaler = amp.GradScaler()
        if self.train_type in ['[Checkpoint]', '[Resume]']:
            self.current_step = (self.start_epoch - 1) * int(self.itr_in_epoch / self.accumulate)
        else:
            self.current_step = 0

    def train(self):
        for epoch in range(self.start_epoch, self.final_epoch + 1):
            self.current_epoch = epoch
            self.model.train()
            self.before_epoch()
            self.print('Epoch: %d/%d'%(self.current_epoch, self.final_epoch))
            time_epoch_start = time.time()
            self.optimizer.zero_grad()
            for i, samples in enumerate(self.train_loader):
                samples['imgs'] = samples['imgs'].to(self.device).float() / 255
                self.normalizer(samples)
                with amp.autocast(enabled=self.using_autocast):
                    loss, loss_dict = self.model(samples)
                loss_log = loss_dict_to_str(loss_dict)
                self.check_loss_or_save(loss)
                self.scaler.scale(loss).backward()
                if self.is_main_process:
                    progressbar((i + 1) / float(self.itr_in_epoch), barlenth=40, endstr=loss_log)
                self.warm_up_setting()
                self.step_and_update(loss_log)
                self.current_step += 1
            time_epoch_end = time.time()
            if self.is_main_process:
                self.save_ckpt('last_epoch')
                self.logger.info('Complete epoch %d at %.1f min, saving last_epoch.pth as last ckpt'
                                 %(self.current_epoch,(time_epoch_end-time_epoch_start)/60))
            self.after_epoch()

    def step_and_update(self, loss_log):
        if not hasattr(self, 'last_step'):
            self.last_step = self.current_step
        if self.accumulate == 1 or self.current_step - self.last_step == self.accumulate:
            if self.is_main_process:
                self.logger_loss.info('epoch ' + str(self.current_epoch) + '/' + str(self.final_epoch) +
                                      ' ' + loss_log)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            if self.ema:
                self.ema.update(self.model)
            self.last_step = self.current_step

    def before_epoch(self):
        if (self.final_epoch + 1 - self.current_epoch) == 50:
            if self.config.training.eval_interval > 5:
                self.print("Begin evaluate on every 5 epoch")
                self.config.training.eval_interval = 5
                self.log_info("Begin evaluate on every 5 epoch")
        if (self.final_epoch + 1 - self.current_epoch) == 20:
            self.print("Begin evaluate on every 1 epoch")
            self.config.training.eval_interval = 1
            self.log_info("Begin evaluate on every 1 epoch")
        if (self.final_epoch + 1 - self.current_epoch) == self.config.training.last_no_mosaic:
            self.print(
                "%d epoches before the end of training, change to no mosaic training." % self.config.training.last_no_mosaic)
            self.change_to_no_mosaic_training()
            self.log_info(
                "%d epoches before the end of training, change to no mosaic training" % self.config.training.last_no_mosaic)
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    def after_epoch(self):
        if self.current_epoch % self.args.save_interval == 0:
            self.save_ckpt('epoch_%d' % self.current_epoch)
            self.log_info("Reach save interval, saving ckpt as epoch_%d.pth" % self.current_epoch)
        if self.current_epoch % self.config.training.eval_interval == 0:
            if self.using_val:
                try:
                    self.valfun()
                except:
                    self.print("Error during eval..")
                    self.log_warn("Error during eval :(")

    def summary(self):
        if self.using_val:
            self.print('Best Val Epochs: %s, %s, %s' % tuple(self.best_epoch_file))
            self.log_info('Best Val Epochs: %s, %s, %s' % tuple(self.best_epoch_file))
            if hasattr(self, 'ap'):
                val_result = self.ap[0]
                self.print('Best AP: %.6f' % val_result)
                self.log_info('Best AP: %.6f' % val_result)
            elif hasattr(self, 'map50'):
                val_result = self.map50[0]
                self.print('Best AP.5: %.6f' % val_result)
                self.log_info('Best AP.5: %.6f' % val_result)

    def warm_up_setting(self):
        if self.current_step <= self.warm_up_steps * self.accumulate:
            # self.accumulate = max(1, np.interp(self.current_step, [0, self.warm_up_steps * self.accumulate * 0.6], [1, self.accumulate]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.config.training.optimizer.warm_up_init_lr
                param['lr'] = np.interp(self.current_step, [0, self.warm_up_steps * self.accumulate],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.current_epoch-1)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(self.current_step, [0, self.warm_up_steps * self.accumulate],
                                                  [self.config.training.optimizer.warm_up_init_momentum, self.config.training.optimizer.momentum])

    def load_finetune_model(self):
        self.print('FineTuning Model: ', end='')
        if self.using_fine_tune:
            self.print(self.args.fine_tune)
            self.print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.fine_tune)
                try:
                    self.model.load_state_dict(ckpt_file['model'])
                except:
                    self.print('FAIL')
                    self.print('\t-Parallel Model Loading:',end=' ')
                    self.model.load_state_dict(de_parallel(ckpt_file['model']))
                self.print('SUCCESS')
                self.log_info('Using FineTuning Model: %s'%self.args.fine_tune)
            except:
                self.print("FAIL")
                raise
        else:
            self.print('None')

    def load_ckpt(self):
        self.print('Checkpoint File:', end=' ')
        if self.using_ckpt:
            self.print(self.args.ckpt_file)
            self.print("\t-Loading:", end=' ')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                try:
                    self.model.load_state_dict(ckpt_file['model'])
                except:
                    self.print('FAIL')
                    self.print('\t-Parallel Model Loading:',end=' ')
                    self.model.load_state_dict(de_parallel(ckpt_file['model']))

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
        else:
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

    def keep_last_ckpt(self, new_file_name = 'special_ckpt'):
        if not self.is_main_process: return
        old_name = os.path.join(self.exp_log_path, 'last_epoch.pth')
        if not os.path.exists(old_name): return
        new_name = os.path.join(self.exp_log_path, new_file_name+'_E%d.pth'%(self.current_epoch-1))
        if os.path.exists(new_name) or new_name == old_name: return
        os.rename(old_name, new_name)
        self.logger.warning('Keep the ckpt of epoch %d to %s'%(self.current_epoch-1,new_name))

    def save_model(self, file_name):
        if self.is_main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'],)
            model_parameters = {}
            model_parameters['last_epoch'] = self.current_epoch
            model_parameters['model'] = self.model.state_dict()
            torch.save(model_parameters, self.exp_log_path+'/'+file_name+'.pth')

    def load_model_to_device(self):
        if self.rank == -1:
            self.model = self.model.to(self.device)
        else:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
            self.print("DDP mode, Using multiGPU")
            self.log_info("DDO mode, Using MultiGPU")
        if self.ema:
            self.ema.ema = self.ema.ema.to(self.device)

    def load_optimizer_to_device(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

    def build_train_dataloader(self):
        self.train_loader = build_dataloader(self.config.training.train_img_anns_path,
                                             self.config.training.train_img_path,
                                             self.config.data,
                                             self.batchsize, self.rank,
                                             self.config.training.workers,
                                             'train')

    def change_to_no_mosaic_training(self):
        del self.train_loader
        self.config.merge_from_file('odcore/data/data_no_mosaic.yaml')
        self.build_train_dataloader()

    def val_setting(self):
        if not self.is_main_process:
            self.using_val = False
            return
        if isinstance(self.config.training.val_img_path, str) and os.path.exists(self.config.training.val_img_path):
            self.using_val = True
            self.val_temp_json = os.path.join(self.exp_log_path, 'temp_predictions.json')
            self.val_log = os.path.join(self.exp_log_path, self.exp_log_name + '_val.log')
            self.val_type = self.config.training.val_metric
            self.best_epoch_file = ['', '', '']
            if self.val_type == 'coco':
                self.valfun = self.val_coco
                self.coco_parse = self.model.get_coco_parser()
                self.print("Using COCO Metric for eval")
            elif self.val_type == 'mr':
                self.valfun = self.val_mr
                self.coco_parse = self.model.get_coco_parser()
                self.print('Using CrowdHuman Metric for eval')
            else:
                raise NotImplementedError('Invalid Evaluation Metric')
        else:
            self.using_val = False
            self.print("Val path not specify or not exist, No eval")
        self.build_val_dataloader()

    def build_val_dataloader(self):
        if self.is_main_process and self.using_val:
            self.val_loader = build_dataloader(self.config.training.val_img_anns_path,
                                               self.config.training.val_img_path,
                                               self.config.data,
                                               self.batchsize, -1, self.config.training.workers,
                                               'val')
        else: self.val_loader = None

    def val_coco(self):
        if not self.is_main_process: return
        self.model.eval()
        if not hasattr(self, 'map'):
            self.map = [0.0, 0.0, 0.0]
        if not hasattr(self, 'map50'):
            self.map50 = [0.0, 0.0, 0.0]
        itr_in_val = len(self.val_loader)
        results = []
        self.print('Begin Evaluation:')
        self.log_info('Evaluation begin at epoch %d'%self.current_epoch)
        time_start = time.time()
        for i, samples in enumerate(self.val_loader):
            samples['imgs'] = samples['imgs'].to(self.device).float() / 255
            self.normalizer(samples)
            with torch.no_grad():
                results.append(self.model(samples))
            progressbar((i + 1) / float(itr_in_val), barlenth=40)
        if not self.using_DDP:
            self.model.get_stats()
        result_for_json = []
        for result in results:
            result_for_json.extend(self.coco_parse(result))
        with open(self.val_temp_json, 'w') as f:
            json.dump(result_for_json, f)
        self.map_, self.map50_ = coco_eval(self.val_temp_json,
                                         self.val_loader.dataset.annotations,
                                         self.val_log,
                                         'Epoch:%s'%str(self.current_epoch))
        time_end = time.time()
        self.print('mAP: %.6f, mAP50: %.6f'%(self.map_, self.map50_))
        self.log_info('Evaluation Complete at %.2f s, mAP: %.6f, mAP50: %.6f'
                      %(time_end-time_start, self.map_, self.map50_))

        for i in range(3):
            if self.map50_ >= self.map50[i]:
                save_name = 'epoch_%d' % self.current_epoch
                if not os.path.exists(os.path.join(self.exp_log_path, save_name+'.pth')):
                    self.save_ckpt(save_name)
                self.map.insert(i, self.map_)
                self.map50.insert(i, self.map50_)
                self.best_epoch_file.insert(i, save_name)

                disuse_name = self.best_epoch_file[3]

                self.map = self.map[:3]
                self.map50 = self.map50[:3]
                self.best_epoch_file = self.best_epoch_file[:3]

                if disuse_name != '':
                    try:
                        os.remove(os.path.join(self.exp_log_path, disuse_name+'.pth'))
                    except:
                        self.log_warn('Error when deleting .pth file %s' % disuse_name)

                self.log_info('New Best Val Epoch: %s, %s, %s' % tuple(self.best_epoch_file))
                break

    def val_mr(self):
        if not self.is_main_process: return
        self.model.eval()
        if not hasattr(self, 'ap'):
            self.ap = [0.0, 0.0, 0.0]
        if not hasattr(self, 'ar'):
            self.ar = [0.0, 0.0, 0.0]
        itr_in_val = len(self.val_loader)
        results = []
        self.print('Begin Evaluation:')
        self.log_info('Evaluation begin at epoch %d'%self.current_epoch)
        time_start = time.time()
        for i, samples in enumerate(self.val_loader):
            samples['imgs'] = samples['imgs'].to(self.device).float() / 255
            self.normalizer(samples)
            with torch.no_grad():
                results.append(self.model(samples))
            progressbar((i + 1) / float(itr_in_val), barlenth=40)
        if not self.using_DDP:
            self.model.get_stats()
        result_for_json = []
        for result in results:
            result_for_json.extend(self.coco_parse(result))
        with open(self.val_temp_json, 'w') as f:
            json.dump(result_for_json, f)
        self.ap_, self.ar_ = gen_eval(self.val_temp_json,
                                      self.val_loader.dataset.annotations,
                                      self.val_log,
                                      'Epoch:%s'%str(self.current_epoch),
                                      eval_type='mr')
        time_end = time.time()
        self.print('AP: %.6f, AR: %.6f'%(self.ap_, self.ar_))
        self.log_info('Evaluation Complete at %.2f s, AP: %.6f, AR: %.6f'
                      %(time_end-time_start, self.ap_, self.ar_))

        for i in range(3):
            if self.ap_ >= self.ap[i]:
                save_name = 'epoch_%d' % self.current_epoch
                if not os.path.exists(os.path.join(self.exp_log_path, save_name+'.pth')):
                    self.save_ckpt(save_name)
                self.ap.insert(i, self.ap_)
                self.ar.insert(i, self.ar_)
                self.best_epoch_file.insert(i, save_name)

                disuse_name = self.best_epoch_file[3]

                self.ap = self.ap[:3]
                self.ar = self.ar[:3]
                self.best_epoch_file = self.best_epoch_file[:3]

                if disuse_name != '':
                    try:
                        os.remove(os.path.join(self.exp_log_path, disuse_name+'.pth'))
                    except:
                        self.log_warn('Error when deleting .pth file %s' % disuse_name)

                self.log_info('New Best Val Epoch: %s, %s, %s' % tuple(self.best_epoch_file))
                break

    def build_optimizer(self):
        if self.config.training.optimizer.mode == 'default':
            self.build_optimizer_default()
        else:
            opt_builder = BuildOptimizer(self.config)
            self.optimizer = opt_builder.build(self.model)

    def build_optimizer_default(self):
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
        lrf_builder = LFScheduler_Step(self.config)
        lf = lrf_builder.get_lr_fun(int(self.itr_in_epoch/self.accumulate))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.lf = lf

    def check_loss_or_save(self,loss):
        if not self.using_loss_protect or not self.is_main_process: return
        if not hasattr(self, 'last_invalid'):
            self.last_invalid = False
            self.invalid_loss_acc = 0
        if torch.isinf(loss) or torch.isnan(loss):
            self.keep_last_ckpt('emergency_save')
            if not self.last_invalid:
                self.last_invalid = True
                self.invalid_loss_acc = 1
            else:
                self.invalid_loss_acc += 1
        else:
            self.last_invalid = False

        if self.invalid_loss_acc == 3:
            self.print('\nInvalid Loss detect multiple times, exit training!')
            self.logger.error('Invalid Loss detect multiple times, exit training!')
            exit()

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def log_info(self,*args,**kwargs):
        if hasattr(self,'logger'):
            self.logger.info(*args,**kwargs)

    def log_warn(self,*args,**kwargs):
        if hasattr(self,'logger'):
            self.logger.warning(*args,**kwargs)

if __name__ == "__main__":
    pass