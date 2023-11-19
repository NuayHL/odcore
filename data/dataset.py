import json
import random
import cv2
import os
import numpy as np
import torch
from pycocotools.coco import COCO
from copy import deepcopy
from torch.utils.data import Dataset
from utils.misc import progressbar

from .data_augment import (
    GeneralAugmenter,
    Resizer,
    RandomAffine,
    LetterBox,
    Mosaic,
    mix_up
)

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
        self.imgPath = imgFilePath
        self.annotations = COCO(annotationPath)
        self.task = task
        self.image_id = self.annotations.getImgIds()
        self.config_data = config_data
        self.lenth = len(self.annotations.imgs)
        self.ignored_input = config_data.ignored_input
        assert self.task in ['train', 'val']
        if task == 'val':
            self.ignored_input = False

        self.random_affine = RandomAffine(config_data)
        self.mosaic = Mosaic(config_data)
        self.general_augment = GeneralAugmenter(config_data)
        self.resizer = Resizer(config_data)
        self.letterbox = LetterBox(config_data, auto=False, scaleup=(self.task == 'train'))

    def __len__(self):
        return self.lenth

    def __getitem__(self, idx):
        '''
        base output:
            sample['img'] = whc np.uint8 img
            sample['anns] = n x (x y w h c) np.float32 img
        '''
        sample = self.load_sample(idx)
        if self.task == 'train' and random.random() < self.config_data.mosaic:
            self.get_mosaic(sample)
            if random.random() < self.config_data.mixup:
                self.get_mixup(sample)
        else:
            self.letterbox(sample)
            if self.task == 'train':
                self.random_affine(sample)

        if self.task == 'train':
            self.general_augment(sample)
        sample["img"] = sample["img"][:,:,::-1]
        sample["img"] = np.ascontiguousarray(sample["img"])
        return sample

    def load_sample(self,idx):
        img, shape, id = self.load_img(idx)
        anns = self.load_anns(idx)
        sample = {"img": img, "anns": anns, "id": id, 'shape':shape}
        self.resizer(sample)
        return sample

    def load_img(self, idx):
        img = self.annotations.imgs[self.image_id[idx]]
        w0, h0, id = img["width"], img["height"], img["id"]
        img = cv2.imread(os.path.join(self.imgPath, img["file_name"]))
        return img, (w0, h0), id

    def load_anns(self, idx):
        anns = self.annotations.imgToAnns[self.image_id[idx]]
        anns = [ann['bbox']+[ann['category_id']] for ann in anns if ann['category_id'] != -1 or self.ignored_input]
        anns = np.array(anns, dtype=np.float32)
        if anns.shape[0] == 0: anns = np.zeros((1,5))
        if self.config_data.annotation_format == 'x1y1wh':
            anns[:, 0] += anns[:, 2] * 0.5
            anns[:, 1] += anns[:, 3] * 0.5
        elif self.config_data.annotation_format == 'x1y1x2y2':
            anns[:, 2] -= anns[:, 0]
            anns[:, 3] -= anns[:, 1]
            anns[:, 0] += anns[:, 2] * 0.5
            anns[:, 1] += anns[:, 3] * 0.5
        elif self.config_data.annotation_format == 'xywh':
            pass
        else:
            raise NotImplementedError('Expect config.data.annotation_format in %s, %s and %s, but got %s.'
                                      %('xywh','x1y1wh','x1y1x2y2',self.config_data.annotation_format))
        return anns

    def get_mosaic(self, sample):
        idxs = random.choices(range(len(self)), k=3)
        random.shuffle(idxs)
        imgs, labels, ws, hs = [sample['img']], [sample['anns']], [sample['img'].shape[1]], [sample['img'].shape[0]]
        for index in idxs:
            sample_i = self.load_sample(index)
            imgs.append(sample_i['img'])
            labels.append(sample_i['anns'])
            ws.append(sample_i['img'].shape[1])
            hs.append(sample_i['img'].shape[0])
        img4, anns4 = self.mosaic(imgs, hs, ws, labels)
        sample['img'] = img4
        sample['anns'] = anns4

    def get_mixup(self, sample):
        sample2 = self.load_sample(random.randint(0, len(self) - 1))
        self.get_mosaic(sample2)
        mix_up(sample, sample2)

    def get_ori_image(self, idx):
        img = self.annotations.imgs[self.image_id[idx]]
        img = cv2.imread(os.path.join(self.imgPath, img["file_name"]))[:,:,::-1]
        return img

    @staticmethod
    def OD_default_collater(data):
        '''
        used in torch.utils.data.DataLaoder as collater_fn
        parse the batch_size data into input_dict
        {"imgs":List lenth B, each with torch.uint8 img
         "anns":List lenth B, each with np.float32 ann, annType: x1y1wh
         "ids": List lenth B, each with str imgid}
        '''
        imgs = torch.stack([torch.from_numpy(np.transpose(s["img"], (2, 0, 1))) for s in data])
        annos = [s["anns"] for s in data]
        ids = [s["id"] for s in data]
        shapes = [s["shape"] for s in data]
        return {"imgs": imgs, "annss": annos, "ids":ids, "shapes":shapes}



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

def check_anno_bbox(coco_style_json):
    coco_dataset = COCO(coco_style_json)
    ori_coco_ann = coco_dataset.dataset
    lenth = float(len(ori_coco_ann["annotations"]))
    for ann in ori_coco_ann["annotations"]:
        its_img = coco_dataset.imgs[ann['image_id']]
        w, h = its_img['width'], its_img['height']
        bbox = ann['bbox']
        x2, y2 = bbox[0]+bbox[2], bbox[1]+bbox[3]
        x2 = x2 if x2<=w else w
        y2 = y2 if y2<=h else h
        ann['bbox'][0] = bbox[0] if bbox[0]>=0 else 0
        ann['bbox'][1] = bbox[1] if bbox[1]>=0 else 0
        ann['bbox'][2] = x2 - bbox[0]
        ann['bbox'][3] = y2 - bbox[1]
        progressbar(ann["id"] / lenth)
    with open(coco_style_json[:-5] + "_checked.json", "w") as f:
        json.dump(ori_coco_ann, f)


class VideoReader():
    def __init__(self, video_path):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.lenth = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info_print()

    def info_print(self):
        print('Video: %s' % self.path)
        print('\t-FPS: %f' % self.fps)
        print('\t-TOTAL FRAMES: %d' % self.lenth)
        print('\t-SIZE: %d X %d' % (self.size[0], self.size[1]))

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.cap.read()
        if not success:
            self.cap.release()
            raise StopIteration
        return frame[:, :, ::-1]

    def __len__(self):
        return self.lenth


class FileImgReader():
    def __init__(self, image_path):
        self.path = image_path
        self.image_list = os.listdir(image_path)
        self.image_list.sort()
        test_img = cv2.imread(os.path.join(self.path, self.image_list[0]))[:, :, ::-1]
        self.size = (test_img.shape[1], test_img.shape[0])
        self.lenth = len(self.image_list)
        self.fps = 25  # temp use---------------------------------------------------------------------------------------
        self.info_print()

    def info_print(self):
        print('Video: %s' % self.path)
        print('\t-FPS: %f' % self.fps)
        print('\t-TOTAL FRAMES: %d' % self.lenth)
        print('\t-SIZE: %d X %d' % (self.size[0], self.size[1]))

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.path, self.image_list[idx]))[:, :, ::-1]
        return img

    def __len__(self):
        return self.lenth