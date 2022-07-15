# from dataset import CocoDataset
#
# dataset = CocoDataset('CrowdHuman/annotation_val_coco_style.json','CrowdHuman/Images_val', ignored_input=True)
# anns = dataset.load_anns(100)
# print(type(anns))
# print(anns)
import numpy as np

def change(a):
    a[0] = 0
a = [1,2]
a = np.array(a)
a_ = {0:a}
print(a)
b=a
change(a_)
print(a)
print(a_)