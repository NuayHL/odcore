from dataset import CocoDataset

dataset = CocoDataset('CrowdHuman/annotation_val_coco_style.json','CrowdHuman/Images_val', ignored_input=True)
anns = dataset.load_anns(100)
print(type(anns))
print(anns)