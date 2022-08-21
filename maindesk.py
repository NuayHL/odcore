import torch

from testtools import *

#
#
# opdict = op.state_dict()
# scdict = sc.state_dict()
# print(opdict.keys())
# print(opdict)
# print(scdict.keys())
# print(scdict)
#
# dataset = CocoDataset('CrowdHuman/annotation_train_coco_style_100.json','CrowdHuman/Images_train', config_data=cfg.data,
#                       task='train')
# dataset_inspection(dataset, 78, anntype='xywh')
# dataset_inspection(dataset, 78, anntype='xywh')
# dataset_inspection(dataset, 78, anntype='xywh')

# samples = [dataset[990],dataset[990],dataset[990],dataset[990]]
# imgs = [sample['img'] for sample in samples]
# labels = [sample['anns'] for sample in samples]
# hs = [640,640,640,640]
# ws = [640,640,640,640]
# #sample['img'],sample['anns']=random_affine(sample['img'],sample['anns'])
# img, label = mosaic_augmentation((800, 1200), imgs, hs, ws, labels, cfg.data)
# show_bbox(img, label, type='x1y1wh')
#
# draw_loss('testing_2.log')

# a = torch.ones((8,3, 5,5))
# b = torch.tensor([[[1]],[[2]],[[3]]])
# c = a-b
# print(c[4,2,:,:])
try:
    a = 1
    try:
        assert a ==2
    except:
        print('lll')
        raise
except:
    print('outer')
print('good')