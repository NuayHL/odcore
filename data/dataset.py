import cv2
import os
import numpy as np
import torch
from pycocotools.coco import COCO
from copy import deepcopy
from torch.utils.data import Dataset

from .data_argment import *

"""
dataset output:
    {'img': }
"""

class CocoDataset(Dataset):
    '''
    Two index system:
        using idx: search using the idx, the position which stored the image_id. Start from 0.
        using id: search using image_id. Usually starts from 1.
    Relation:
        Default: id = idx + 1
        Always right: id = self.image_id[idx] (adopted)
    Annotation format:
        x1y1wh
    '''
    def __init__(self, annotationPath, imgFilePath, config_data, task='train'):
        super(CocoDataset, self).__init__()
        self.jsonPath = annotationPath
        self.imgPath = imgFilePath + "/"
        self.annotations = COCO(annotationPath)
        self.image_id = self.annotations.getImgIds()
        self.task = task
        self.config_data = config_data
        self.ignored_input = config_data.ignored_input
        assert self.task in ['train', 'eval']
        if task == 'eval':
            self.ignored_input = False

        self.normalizer = Normalizer(config_data)
        self.general_argm = GeneralAugmenter(config_data)
        self.resizer = Resizer(config_data)

    def __len__(self):
        return len(self.annotations.imgs)

    def __getitem__(self, idx):
        '''
        base output:
            sample['img'] = whc np.int32? img
            sample['anns] = n x (x1 y1 w h c) np.int32 img
        '''
        img, (w0, h0), img_id = self.load_img(idx)
        anns = self.load_anns(idx)
        sample = {"img":img, "anns":anns, "id":img_id}
        self.resizer(sample, w0, h0)
        if self.task == 'train' and np.random.rand() < self.config_data.mosaic:
            pass
        self.general_argm(sample)
        sample["img"] = sample["img"][:,:,::-1]
        #sample["img"] = sample["img"].transpose((2,0,1))[:,:,::-1]
        sample["img"] = np.ascontiguousarray(sample["img"])
        return sample

    def load_img(self, idx):
        img = self.annotations.imgs[self.image_id[idx]]
        w0, h0, id = img["width"], img["height"], img["id"]
        img = cv2.imread(self.imgPath + img["file_name"] + ".jpg")
        return img, (w0, h0), id

    def load_anns(self, idx):
        anns = self.annotations.imgToAnns[self.image_id[idx]]
        anns = [ann['bbox']+[ann['category_id']] for ann in anns if ann['category_id'] != -1 or self.ignored_input]
        anns = np.array(anns, dtype=np.float32)
        return anns

    def get_mosaic(self, idx):
        pass

    @staticmethod
    def OD_default_collater(data):
        '''
        used in torch.utils.data.DataLaoder as collater_fn
        parse the batch_size data into dict
        {"imgs":List lenth B, each with np.uint8 img
         "anns":List lenth B, each with np.float32 ann, annType: x1y1wh}
        '''
        imgs = torch.stack([torch.from_numpy(np.transpose(s["img"], (2, 0, 1))) for s in data])
        annos = [s["anns"] for s in data]
        ids = [s["id"] for s in data]
        return {"imgs": imgs, "anns": annos, "ids":ids}

class MixCocoDatset(Dataset):
    """
    Used for combining different dataset together.
    """
    def __init__(self, datasets: list[CocoDataset]):
        super(MixCocoDatset, self).__init__()
        self.cocodataset = datasets
        self.divids = [0]
        for i, dataset in enumerate(self.cocodataset):
            self.divids.append(len(dataset)+self.divids[i])

    def addDataset(self, dataset:CocoDataset):
        self.cocodataset.append(dataset)
        self.divids.append(len(dataset)+self.divids[-1])

    def __len__(self):
        lenth = 0
        for dataset in self.cocodataset:
            lenth += len(dataset)
        return lenth

    def __getitem__(self, idx):
        for i in range(len(self.divids)-1):
            if idx>=self.divids[i] and idx<self.divids[i+1]:
                return self.cocodataset[i][idx-self.divids[i]]
        return self.cocodataset[-1][idx-self.divids[-1]]

class Txtdatset(Dataset):
    """
    anns format:
        n x 5: n x (xywh c) np.float32
    """
    def __init__(self, imgs_addr, imgs_label_addr, transform=None, xywhoutput=True):
        self.imgs_addr = imgs_addr
        self.imgs_label_addr = imgs_label_addr
        self.imgs = os.listdir(imgs_addr)
        self.xywhoutput = xywhoutput
        self.transform = transform

    def __getitem__(self, idx):
        imgname = self.imgs[idx]
        img = cv2.imread(self.imgs_addr + "/" + imgname)
        img = img[:, :, ::-1]
        w, h = img.shape[1], img.shape[0]
        with open(self.imgs_label_addr+"/"+self.imgs[idx][:-4]+".txt", "r") as f:
            bboxes = f.readlines()
        bboxes = [bbox.split() for bbox in bboxes]
        bboxes = [[float(num) for num in bbox] for bbox in bboxes]
        bboxes = [bbox[1:]+[bbox[0]] for bbox in bboxes]
        anns = np.array(bboxes, dtype=np.float32)
        anns[:, 0] *= w
        anns[:, 2] *= w
        anns[:, 1] *= h
        anns[:, 3] *= h
        if not self.xywhoutput:
            self._xywh_to_x1y1wh(anns)
        sample = {"img":img, "anns":anns}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _xywh_to_x1y1wh(self, anns):
        pass

def load_single_inferencing_img(img, device):
    '''
    Used for inferencing one single img
    :param img:
        str: file path
        np.ndarray: W x H x C
        torch.Tensor: B x C x W x H
    :return: Input Tensor viewed as batch_size 1
    '''
    if isinstance(img,str):
        img = cv2.imread(img)
        img = img[:,:,::-1]
    elif isinstance(img,torch.Tensor):
        if torch.cuda.is_available():
            return img.to(device)
        else:
            return img
    elif isinstance(img,np.ndarray):
        pass
    else:
        raise NotImplementedError("Unknown inputType")

    img = (cv2.resize(img.astype(np.float32), (cfg.input_width, cfg.input_height)))/255
    img = np.transpose(img,(2,0,1))
    img = preprocess_train(torch.from_numpy(img).float())
    img = torch.unsqueeze(img, dim=0)
    return img.to(device)


