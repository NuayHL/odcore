import torch
import torch.nn

class Infer():
    def __init__(self, config, args, model, device):
        self.config = config
        self.args = args
        self.model = model
        self.device = device
        self.load_model()

    def load_model(self):
        self.print('FineTuning Model: ', end='')
        if self.args.fine_tune != '':
            self.print(self.args.fine_tune)
            self.print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.fine_tune)
                self.model.load_state_dict(ckpt_file['model'])
                self.print('SUCCESS')
                self.logger.info('Using FineTuning Model: %s'%self.args.fine_tune)
            except:
                self.print("FAIL")
                raise
            self.model = self.model.to(self.device)
        else:
            self.print('Please indicating one .pth/.pt file!')
            exit()

    def __call__(self, img):
        
