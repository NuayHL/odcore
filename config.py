import yaml
from yacs.config import CfgNode as _CN

class CN(_CN):
    def __init__(self):
        super(CN, self).__init__()

    def dump_to_file(self):
        cfg_string = self.dump()
        assert hasattr(self, 'name')
        with open(self.name + '.yaml', "w") as f:
            f.write(cfg_string)


c = CN()
c.name = 'yolov3'

c.model = CN()
c.model.backbone = 'darknet53'
c.model.neck = 'yolov3'
c.model.detector = 'standard_yolo'

c.data = CN()
c.data.input_mean = [0.46431773, 0.44211456, 0.4223358]
c.data.input_std = [0.29044453, 0.28503336, 0.29363019]
c.data.input_width = 640
c.data.input_height = 640
c.data.ignored_input = True
c.data.numofclasses = 1

c.data.hsv_h = 0.0138
c.data.hsv_s = 0.664
c.data.hsv_v = 0.464
c.data.degrees = 0.373
c.data.translate = 0.245
c.data.scale = 0.898
c.data.shear = 0.602
c.data.flipud = 0.00856
c.data.fliplr = 0.5
c.data.mosaic = 1.0
c.data.mixup = 0.243

c.training = CN()

c.training.use_anchor = False
# if using anchor
c.training.fpnlevels = [3, 4, 5]
c.training.ratios = [2, 4]
c.training.scales = [0.75, 1]
# if not using anchor

c.training.assignment = 'default'

c.training.train_img_path = ''
c.training.train_img_anns_path = ''
c.training.val_img_path = ''
c.training.val_img_anns_path = ''
c.training.batch_size = 8

c.training.optimizer = CN()
c.training.optimizer.type = 'SGD'
c.training.optimizer.lr = 0.001
c.training.optimizer.weight_decay = 0.001
c.training.optimizer.momentum = 0.001       #SGD
c.training.optimizer.betas = (0.9, 0.999)   #AdamW

c.training.loss = CN()
c.training.loss.reg_type = ['giou','l1']
c.training.loss.cls_type = 'bce'
c.training.loss.usefocal = False


c.inference = CN()

def get_default_cfg():
    return c.clone()

def get_default_yaml_templete():
    cfg_string = get_default_cfg().dump()
    with open("default_config.yaml", "w") as f:
        f.write(cfg_string)

if __name__ == "__main__":
    get_default_yaml_templete()