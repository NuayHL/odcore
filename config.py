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
c.model.detector = 'yolov3'

c.data = CN()
c.data.input_mean = [0.46431773, 0.44211456, 0.4223358]
c.data.input_std = [0.29044453, 0.28503336, 0.29363019]
c.data.input_width = 640
c.data.input_height = 640
c.data.ignored_input = True

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
c.training.regloss = ['siou', 'l1']
c.training.clsloss = 'bce'
c.training.usefocal = False

c.training.train_img_path = ''
c.training.train_img_anns_path = ''
c.training.val_img_path = ''
c.training.val_img_anns_path = ''

c.inference = CN()

def get_default_cfg():
    return c.clone()

def get_default_yaml_templete():
    cfg_string = get_default_cfg().dump()
    with open("default_config.yaml", "w") as f:
        f.write(cfg_string)

if __name__ == "__main__":
    get_default_yaml_templete()