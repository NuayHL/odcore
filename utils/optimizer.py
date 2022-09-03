import torch
import torch.nn as nn
import torch.optim as optim

class BuildOptimizer():
    default_para_groups = ['backbone','neck','head']
    def __init__(self, config):
        self.config_opt = config.training.optimizer

    def get_para_groups(self):
        self.para_groups = []

    def build(self, model: nn.Module):
        para_groups = {}
        if self.config_opt.mode == 'none':
            para_groups['all'] = model.parameters()
        else:
            if self.config_opt.mode == 'groups':
                for part in self.default_para_groups:
                    if hasattr(model, part):
                        para_groups.append(model.__getattribute__(part).parameters())
                for part in self.config_opt.para_group:
                    if hasattr(model, part):
                        para_groups[part] = model.__getattribute__(part).parameters()


            elif self.config_opt.mode == 'types':
                for layer in model.modules():





        if self.config_opt.type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(),
                                       lr = self.config_opt.lr,
                                       momentum=self.config_opt.momentum, nesterov=True)
        elif self.config_opt.type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(),
                                        lr = self.config_opt.lr,
                                        betas=(self.config_opt.momentum, 0.999))

        pass


