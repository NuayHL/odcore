import math

class LFScheduler():
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
            list = self.lr_dict['milestones']
            ratio = self.lr_dict['ratio']
            def step_lr(x):
                for idx in range(len(list)):
                    if x < list[idx]: return ratio ** idx
                return ratio ** len(list)
            self.lf = step_lr
        else:
            raise NotImplementedError

        return self.lf

