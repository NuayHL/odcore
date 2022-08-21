import os
import torch
from utils.visualization import LossLog

class Exp():
    def __init__(self, exp_file_path, is_main_process):
        self.exp_path = exp_file_path
        self.is_main_process = is_main_process
        self.check_exp()

    def check_exp(self):
        self.print('Checking Path Validity:',os.path.exists(self.exp_path))
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError('%s do not exit!'%self.exp_path)
        self.exp_name = os.path.basename(self.exp_path)
        check_flag = True
        files = os.listdir(self.exp_path)
        self.args_file_name = self.exp_name+'_args.yaml'
        self.cfg_file_name = self.exp_name+'_cfg.yaml'
        self.log_file_name = self.exp_name+'.log'
        self.log_loss_file_name = self.exp_name+'_loss.log'
        if self.args_file_name in files:
            self.print('\t-'+self.args_file_name + ' find')
        else:
            self.print('\t-'+self.args_file_name + ' not find')
            self.args_file_name = False

        if self.cfg_file_name in files:
            self.print('\t-'+self.cfg_file_name + ' find')
        else:
            self.print('\t-'+self.cfg_file_name + ' not find')
            check_flag = False

        if self.log_file_name in files:
            self.print('\t-'+self.log_file_name + ' find')
        else:
            self.print('\t-'+self.log_file_name + ' not find')
            self.log_file_name = False

        if self.log_loss_file_name in files:
            self.print('\t-'+self.log_loss_file_name + ' find')
        else:
            self.print('\t-Warning: '+self.log_loss_file_name + ' not find')
            self.log_loss_file_name = False

        if 'last_epoch.pth' in files or 'best_epoch.pth' in files:
            self.print('\t-'+'Checkpoint files find:', end=' ')
            if 'last_epoch.pth' in files:
                self.print('last_epoch.pth')
                self.ckpt_file_name = 'last_epoch.pth'
            else:
                self.print('best_epoch.pth')
                self.ckpt_file_name = 'best_epoch.pth'
        else:
            self.print('\t-'+'Neither Checkpoint files find')
            check_flag = False

        if check_flag == False:
            self.print('Exp file incomplete')
            raise FileNotFoundError('Exp file incomplete')
        self.files = files
        if self.log_loss_file_name:
            self.print('Checking Loss Log...')
            self.log_loss_file = LossLog(os.path.join(self.exp_path,self.log_loss_file_name),
                                         self.is_main_process)

    def get_cfg_path(self):
        return os.path.join(self.exp_path,self.cfg_file_name)

    def get_ckpt_file_path(self):
        return os.path.join(self.exp_path, self.ckpt_file_name)

    def get_exp_files(self):
        return self.files

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)
