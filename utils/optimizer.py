import torch
import torch.nn as nn
import torch.optim as optim
import warnings

class BuildOptimizer():
    default_para_groups = ['backbone','neck','head']
    default_para_types = ['weight','bias','batch_norm']
    def __init__(self, config):
        self.config_opt = config.training.optimizer

    def build(self, model: nn.Module):
        self.para_groups = {}
        try:
            self.defined_groups = self.config_opt.para_group[0].keys()
        except:
            if self.config_opt.mode != 'none':
                warnings.warn('Not define para groups, Please change to other mode or specify para groups!')
            self.defined_groups = []

        if self.config_opt.mode == 'none':
            self.para_groups['all'] = model.parameters()
        else:
            if self.config_opt.mode == 'groups':
                for part in self.default_para_groups:
                    if hasattr(model, part):
                        self.para_groups[part] = model.__getattr__(part).parameters()
                for part in self.defined_groups:
                    if hasattr(model, part) and part not in self.default_para_groups:
                        self.para_groups[part] = model.__getattr__(part).parameters()

            elif self.config_opt.mode == 'types':
                for part in self.default_para_types:
                    self.para_groups[part] = []
                for v in model.modules():
                    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                        self.para_groups['bias'].append(v.bias)
                    if isinstance(v, nn.BatchNorm2d):
                        self.para_groups['batch_norm'].append(v.weight)
                    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                        self.para_groups['weight'].append(v.weight)
            else:
                raise NotImplementedError

        para_keys = self.para_groups.keys()
        for index,part in enumerate(para_keys):
            if self.config_opt.type.lower() == 'sgd':
                part_lr = self.config_opt.lr
                part_momentum = self.config_opt.momentum
                if part in self.defined_groups:
                    part_lr *= self.config_opt.para_group[0][part]['lr']
                    part_momentum *= self.config_opt.para_group[0][part]['momentum']

                if index == 0:
                    self.optimizer = optim.SGD(self.para_groups[part],
                                               lr = part_lr,
                                               momentum=part_momentum, nesterov=True)
                else:
                    self.optimizer.add_param_group({'params': self.para_groups[part], 'lr': part_lr,
                                                    'momentum': part_momentum})

            elif self.config_opt.type.lower() == 'adam':
                part_lr = self.config_opt.lr
                part_momentum = self.config_opt.momentum
                if part in self.defined_groups:
                    part_lr *= self.config_opt.para_group[0][part]['lr']
                    part_momentum *= self.config_opt.para_group[0][part]['momentum']
                if index == 0:
                    self.optimizer = optim.Adam(self.para_groups[part],
                                                lr = part_lr,
                                                betas=(part_momentum, 0.999))
                else:
                    self.optimizer.add_param_group({'params': self.para_groups[part],
                                                    'lr': part_lr,
                                                    'betas':(part_momentum, 0.999)})

            else:
                raise NotImplementedError

        return self.optimizer

