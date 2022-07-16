from data.dataset import CocoDataset
from visualization import dataset_inspection
from config import get_default_cfg
cfg = get_default_cfg()

dataset = CocoDataset('CrowdHuman/annotation_val_coco_style.json','CrowdHuman/Images_val', config_data=cfg.data)
dataset_inspection(dataset, 1000)
dataset_inspection(dataset, 1000)
dataset_inspection(dataset, 1000)
dataset_inspection(dataset, 1000)
dataset_inspection(dataset, 1000)

