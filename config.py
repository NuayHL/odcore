import yaml
from yacs.config import CfgNode as _CN

class CN(_CN):
    def __init__(self):
        super(CN, self).__init__()

c = CN()
c.model = CN()
c.model.backbone = 'darknet53'
c.model.neck = 'yolov3'
c.model.detector = 'yolov3'

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
    cfg = get_default_cfg()
    cfg.merge_from_file('default_config.yaml')
    print(cfg)