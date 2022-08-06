import os
import torch

class Exp():
    def __init__(self, exp_file_path):
        self.exp_path = exp_file_path
        self.check_exp()

    def check_exp(self):
        print('Checking Path Validity:',os.path.exists(self.exp_path))
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError('%s do not exit!'%self.exp_path)
        self.exp_name = os.path.basename(self.exp_path)
        check_flag = True
        files = os.listdir(self.exp_path)
        self.args_file_name = self.exp_name+'_args.yaml'
        self.cfg_file_name = self.exp_name+'_cfg.yaml'
        self.log_file_name = self.exp_name+'.log'
        if self.args_file_name in files:
            print('\t-'+self.args_file_name + ' find')
        else:
            print('\t-'+self.args_file_name + ' not find')
            check_flag = False

        if self.cfg_file_name in files:
            print('\t-'+self.cfg_file_name + ' find')
        else:
            print('\t-'+self.cfg_file_name + ' not find')
            check_flag = False

        if self.log_file_name in files:
            print('\t-'+self.log_file_name + ' find')
        else:
            print('\t-'+self.log_file_name + ' not find')
            check_flag = False

        if 'last_epoch.pth' in files or 'best_epoch.pth' in files:
            print('\t-'+'Checkpoint files find:', end=' ')
            if 'last_epoch.pth' in files:
                print('last_epoch.pth')
                self.ckpt_file_name = 'last_epoch.pth'
            else:
                print('best_epoch.pth')
                self.ckpt_file_name = 'best_epoch.pth'
        else:
            print('\t-'+'Neither Checkpoint files find')
            check_flag = False

        if check_flag == False:
            print('Exp file incomplete')
            raise FileNotFoundError('Exp file incomplete')

    def get_cfg_path(self):
        return os.path.join(self.exp_path,self.cfg_file_name)
