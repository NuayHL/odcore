import torch
import torch.nn

class Infer():
    def __init__(self, config, args, model):
        self.config = config
        self.args = args
        self.model = model

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
        else:
            self.print('None')


    def __call__(self, img):
        if
