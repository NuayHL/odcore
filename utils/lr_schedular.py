import math

class LFScheduler:
    def __init__(self, config):
        self.no_mosaic_epoch = config.training.last_no_mosaic
        self.final_epoch = config.training.final_epoch
        self.config_lr = config.training.schedular
        self.lr_type = self.config_lr.type
        self.lr_dict = self.config_lr.extra[0]

    def get_lr_fun(self):
        self.lf = lambda x:1.0
        if self.lr_type == 'cosine':
            lrf = self.lr_dict['lrf']
            def cosine(x):
                if x <= self.final_epoch:
                    return ((1 - math.cos(x * math.pi / self.final_epoch)) / 2) * (lrf - 1) + 1
                else:
                    return lrf
            self.lf = cosine
            #self.lf = lambda x: ((1 - math.cos(x * math.pi / self.final_epoch)) / 2) * (lrf - 1) + 1
        elif self.lr_type == 'detail_cosine':
            lrf = self.lr_dict['lrf']
            def detail_cosine(x):
                if x<= self.final_epoch-self.no_mosaic_epoch:
                    return ((1 - math.cos(x * math.pi / (self.final_epoch-self.no_mosaic_epoch))) / 2) * (lrf - 1) + 1
                else:
                    return lrf
            self.lf = detail_cosine
        elif self.lr_type == 'step':
            mlist = self.lr_dict['milestones']
            ratio = self.lr_dict['ratio']
            def step_lr(x):
                for idx in range(len(list)):
                    if x < mlist[idx]: return ratio ** idx
                return ratio ** len(list)
            self.lf = step_lr
        elif self.lr_type == 'poly':
            mlist = [0] + self.lr_dict['milestones']
            ratios = [1.0] + self.lr_dict['ratios']
            assert len(mlist) == len(ratios)
            def poly_lr(x):
                for id in range(len(mlist)):
                    if x >= mlist[id]:
                        if id == len(mlist) - 1:
                            return ratios[-1]
                        elif x < mlist[id + 1]:
                            return ratios[id] + (ratios[id+1] - ratios[id]) / (mlist[id+1] - mlist[id]) * (x - mlist[id])
                        else:
                            continue
            self.lf = poly_lr
        elif self.lr_type == 'linear':
            lrf = self.lr_dict['lrf']
            def linear(x):
                return 1.0 - x * (1.0 - lrf) / self.final_epoch
            self.lf = linear
        else:
            raise NotImplementedError

        return self.lf

class LFScheduler_Step:
    def __init__(self, config):
        self.no_mosaic_epoch = config.training.last_no_mosaic
        self.final_epoch = config.training.final_epoch
        self.config_lr = config.training.schedular
        self.lr_type = self.config_lr.type
        self.lr_dict = self.config_lr.extra[0]
        self.lf = lambda x: 1.0

    def get_lr_fun(self, iter_in_epoch):
        if self.lr_type == 'cosine':
            lrf = self.lr_dict['lrf']
            def cosine(x):
                fin_steps = self.final_epoch * iter_in_epoch
                if x <= fin_steps:
                    return ((1 - math.cos(x * math.pi / fin_steps)) / 2) * (lrf - 1) + 1
                else:
                    return lrf
            self.lf = cosine
            #self.lf = lambda x: ((1 - math.cos(x * math.pi / self.final_epoch)) / 2) * (lrf - 1) + 1
        elif self.lr_type == 'detail_cosine':
            lrf = self.lr_dict['lrf']
            def detail_cosine(x):
                fin_steps = (self.final_epoch-self.no_mosaic_epoch) * iter_in_epoch
                if x <= fin_steps:
                    return ((1 - math.cos(x * math.pi / fin_steps)) / 2) * (lrf - 1) + 1
                else:
                    return lrf
            self.lf = detail_cosine
        elif self.lr_type == 'step':
            mlist = self.lr_dict['milestones']
            ratio = self.lr_dict['ratio']
            def step_lr(x):
                for idx in range(len(list)):
                    if x < mlist[idx] * iter_in_epoch: return ratio ** idx
                return ratio ** len(list)
            self.lf = step_lr
        elif self.lr_type == 'poly':
            mlist = [0] + self.lr_dict['milestones']
            ratios = [1.0] + self.lr_dict['ratios']
            assert len(mlist) == len(ratios)
            mlist = [x * iter_in_epoch for x in mlist]
            def poly_lr(x):
                for id in range(len(mlist)):
                    if x >= mlist[id]:
                        if id == len(mlist) - 1:
                            return ratios[-1]
                        elif x < mlist[id + 1]:
                            return ratios[id] + (ratios[id+1] - ratios[id]) / (mlist[id+1] - mlist[id]) * (x - mlist[id])
                        else:
                            continue
            self.lf = poly_lr
        elif self.lr_type == 'linear':
            lrf = self.lr_dict['lrf']
            def linear(x):
                fin_steps = self.final_epoch * iter_in_epoch
                return 1.0 - x * (1.0 - lrf) / fin_steps
            self.lf = linear
        else:
            raise NotImplementedError

        return self.lf


