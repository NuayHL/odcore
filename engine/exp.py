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
        if self.exp_name+'_args.yaml' in files:
            print(self.exp_name+'_args.yaml find')
        else:
            print(self.exp_name + '_args.yaml not find')
            check_flag = False

        if self.exp_name+'_cfg.yaml' in files:
            print(self.exp_name+'_cfg.yaml find')
        else:
            print(self.exp_name + '_cfg.yaml not find')
            check_flag = False

        if self.exp_name+'.log' in files:
            print(self.exp_name+'.log find')
        else:
            print(self.exp_name + '.log not find')
            check_flag = False

        if 'last_epoch.pth' in files or 'best_epoch.pth' in files:
            print('Checkpoint files find')
        else:
            print('Neither Checkpoint files not find')
            check_flag = False

        if check_flag == False:
            print('Exp file incomplete')
            raise FileNotFoundError('Exp file incomplete')
