from testtools import *


model = DummyModel().cuda()
op = optim.SGD(model.parameters(),lr=1)
sc = scheduler.MultiStepLR(op,[5,10],gamma=0.1)
op.zero_grad()


opdict = op.state_dict()
scdict = sc.state_dict()
print(opdict.keys())
print(opdict)
print(scdict.keys())
print(scdict)

# dataset = CocoDataset('CrowdHuman/annotation_val_coco_style.json','CrowdHuman/Images_val', config_data=cfg.data,
#                       task='train')
# dataset_inspection(dataset, 1110, anntype='x1y1wh')

# samples = [dataset[990],dataset[990],dataset[990],dataset[990]]
# imgs = [sample['img'] for sample in samples]
# labels = [sample['anns'] for sample in samples]
# hs = [640,640,640,640]
# ws = [640,640,640,640]
# #sample['img'],sample['anns']=random_affine(sample['img'],sample['anns'])
# img, label = mosaic_augmentation((800, 1200), imgs, hs, ws, labels, cfg.data)
# show_bbox(img, label, type='x1y1wh')

