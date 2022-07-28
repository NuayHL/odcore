from data.dataset import CocoDataset
from utils.visualization import dataset_inspection, show_bbox
from utils.misc import *
from config import get_default_cfg
from data.data_augment import *
cfg = get_default_cfg()

dataset = CocoDataset('CrowdHuman/annotation_val_coco_style.json','CrowdHuman/Images_val', config_data=cfg.data,
                      task='val')
dataset_inspection(dataset, 991, anntype='x1y1wh')

# samples = [dataset[990],dataset[990],dataset[990],dataset[990]]
# imgs = [sample['img'] for sample in samples]
# labels = [sample['anns'] for sample in samples]
# hs = [640,640,640,640]
# ws = [640,640,640,640]
# #sample['img'],sample['anns']=random_affine(sample['img'],sample['anns'])
# img, label = mosaic_augmentation((800, 1200), imgs, hs, ws, labels, cfg.data)
# show_bbox(img, label, type='x1y1wh')

