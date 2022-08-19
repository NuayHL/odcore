import yaml
import os
from yacs.config import CfgNode as _CN

class CN(_CN):
    # def dump_to_file(self, yaml_name=None, path=''):
    #     cfg_string = self.dump()
    #     if yaml_name is None:
    #         assert hasattr(self, 'exp_name')
    #         file_name = self.exp_name
    #     else:
    #         file_name = yaml_name
    #     with open(os.path.join(path,file_name + '.yaml'), "w") as f:
    #         f.write(cfg_string)
    def dump_to_file(self, yaml_name=None, path=''):
        if yaml_name is None:
            assert hasattr(self, 'exp_name')
            file_name = self.exp_name
        else:
            file_name = yaml_name
        with open(os.path.join(path,file_name + '.yaml'), "w") as f:
            print(self, file=f)

c = CN()
c.exp_name = 'yolov3'

c.data = CN()
c.data.input_mean = [0.46431773, 0.44211456, 0.4223358]
c.data.input_std = [0.29044453, 0.28503336, 0.29363019]
c.data.input_width = 640
c.data.input_height = 640
c.data.ignored_input = True
c.data.numofclasses = 1
c.data.annotation_format = 'x1y1wh'
c.data.hsv_h = 0.0138
c.data.hsv_s = 0.664
c.data.hsv_v = 0.464
c.data.degrees = 0.373
c.data.translate = 0.245
c.data.scale = 0.7
c.data.shear = 0.602
c.data.flipud = 0.00856
c.data.fliplr = 0.5
c.data.mosaic = 1.0
c.data.mixup = 0.243

c.training = CN()

c.training.train_img_path = 'train_img_path'
c.training.train_img_anns_path = 'train_img_anns_path'
c.training.val_img_path = 'val_img_path'
c.training.val_img_anns_path = 'val_img_anns_path'
c.training.val_metric = 'coco'
c.training.batch_size = 8
c.training.final_epoch = 200
c.training.workers = 4
c.training.eval_interval = 20
c.training.using_autocast = True
c.training.warm_up_steps = 1000
c.training.accumulate = 1

c.training.optimizer = CN()
c.training.optimizer.type = 'SGD'
c.training.optimizer.lr = 0.01
c.training.optimizer.weight_decay = 0.0005
c.training.optimizer.momentum = 0.937       #SGD

c.training.schedular = CN()
c.training.schedular.type = 'cosine'
c.training.schedular.lrf = 0.01

def get_default_cfg():
    return c.clone()

def get_default_yaml_templete():
    cfg_string = get_default_cfg().dump()
    with open("default_config.yaml", "w") as f:
        f.write(cfg_string)

if __name__ == "__main__":
    # cfg = get_default_cfg()
    # cfg.merge_from_file('default_config.yaml')
    # print(cfg)
    get_default_yaml_templete()