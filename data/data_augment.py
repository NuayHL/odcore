import numpy as np
import cv2
import random
import math
from utils.misc import *

class Normalizer():
    def __init__(self, config_data):
        self.mean = np.array(config_data.input_mean)
        self.std = np.array(config_data.input_std)
    def __call__(self, sample):
        sample['img'] /= 255
        sample['img'] = (sample['img'] - self.mean)/self.std

class GeneralAugmenter():
    def __init__(self, config_data):
        self.fliplr = config_data.fliplr
        self.flipud = config_data.flipud
        self.hsv_h = config_data.hsv_h
        self.hsv_s = config_data.hsv_s
        self.hsv_v = config_data.hsv_v
    def __call__(self, sample):
        # horizontal flip
        if random.random() < self.fliplr:
            sample["img"] = sample["img"][:,::-1,:]

            _, width, _ = sample["img"].shape
            sample["anns"][:, 0] = width - sample["anns"][:, 0] - sample["anns"][:, 2]
            sample["img"] = np.ascontiguousarray(sample["img"])

        # up-down flip
        if random.random() < self.flipud:
            sample["img"] = sample["img"][::-1,:,:]

            height, _, _ = sample["img"].shape
            sample["anns"][:, 1] = height - sample["anns"][:, 1] - sample["anns"][:, 3]
            sample["img"] = np.ascontiguousarray(sample["img"])

        # HSV color-space augmentation
        if self.hsv_h or self.hsv_s or self.hsv_v:
            r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(sample["img"], cv2.COLOR_BGR2HSV))
            dtype = sample["img"].dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, sample["img"])

class LetterBox():
    def __init__(self, config_data, color=(114,114,114),auto=False, scaleup=True, stride=32):
        self.width = config_data.input_width
        self.height = config_data.input_height
        self.color = color
        self.auto = auto
        self.scaleup = scaleup
        self.stride = stride
    def __call__(self, sample):
        # Resize and pad image while meeting stride-multiple constraints
        shape = sample['img'].shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.height / shape[0], self.width / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.width - new_unpad[0], self.height - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            sample['img'] = cv2.resize(sample['img'], new_unpad, interpolation=cv2.INTER_LINEAR)
            sample["anns"][:, :4] *= r
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        sample['img'] = cv2.copyMakeBorder(sample['img'], top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        sample["anns"][:, 0] += dw
        sample["anns"][:, 1] += dh

class Resizer():
    def __init__(self, config_data):
        self.width = config_data.input_width
        self.height = config_data.input_height
    def __call__(self, sample):
        fy = self.height / float(sample['shape'][1])
        fx = self.width / float(sample['shape'][0])
        f = min(fx, fy)
        sample["anns"][:, 0:4] *= f
        if f != 1:
            sample['img'] = cv2.resize(
                sample['img'],
                (int(float(sample['shape'][0]) * f), int(float(sample['shape'][1]) * f)),
                interpolation=cv2.INTER_AREA
                if f < 1 else cv2.INTER_LINEAR)

class RandomAffine():
    def __init__(self, config_data):
        self.degrees = config_data.degrees
        self.scale = config_data.scale
        self.shear = config_data.shear
        self.translate = config_data.translate
        self.img_size = (config_data.input_width, config_data.input_height)
    def __call__(self, sample):
        n = len(sample['anns'])
        M, s = get_transform_matrix(sample['img'].shape[:2],
                                    (self.img_size[1], self.img_size[0]),
                                    self.degrees,
                                    self.scale,
                                    self.shear,
                                    self.translate)
        if (M != np.eye(3)).any():  # image changed
            sample['img'] = cv2.warpAffine(sample['img'], M[:2], dsize=self.img_size, borderValue=(114, 114, 114))

        # Transform label coordinates
        if n:
            x1y1wh_x1y1x2y2_(sample['anns'])
            xy = np.ones((n * 4, 3))
            xy[:, :2] = sample['anns'][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, self.img_size[0])
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, self.img_size[1])

            # filter candidates
            i = box_candidates(box1=sample['anns'][:, 0:4].T * s, box2=new.T, area_thr=0.1)
            sample['anns'] = sample['anns'][i]
            sample['anns'][:, 0:4] = new[i]

        x1y1x2y2_x1y1wh_(sample['anns'])

class Mosaic():
    def __init__(self, config_data):
        self.img_size = (config_data.input_width, config_data.input_height)
        self.random_affine = RandomAffine(config_data)

    def __call__(self,imgs, hs, ws, labels):
        '''img_size: w, h'''
        assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."
        labels4 = []
        xc, yc = (int(random.uniform(self.img_size[0] // 2, 3 * self.img_size[0] // 2)),
                  int(random.uniform(self.img_size[1] // 2, 3 * self.img_size[1] // 2)))  # mosaic center x, y
        img4 = np.full((self.img_size[1] * 2, self.img_size[0] * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        for i in range(4):
            # Load image
            img, h, w = imgs[i], hs[i], ws[i]
            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (out image location)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (crop ori image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size[0] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size[0] * 2), min(self.img_size[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            print(padw)
            # Labels
            labels[i][:, 0] += padw  # top left x
            labels[i][:, 1] += padh  # top left y
            boxes = x1y1wh_x1y1x2y2(labels[i])
            labels4.append(boxes)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        labels4[:, [0, 2]] = labels4[:, [0, 2]].clip(0, 2 * self.img_size[0])
        labels4[:, [1, 3]] = labels4[:, [1, 3]].clip(0, 2 * self.img_size[1])
        x1y1x2y2_x1y1wh_(labels4)
        sim_sample = {'img':img4, 'anns':labels4}
        print('mo_1:', len(sim_sample['anns']), end='\t')
        # Augment
        #self.random_affine(sim_sample)
        print('mo_r:', len(sim_sample['anns']), end='\t')
        return sim_sample['img'], sim_sample['anns']

def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def mix_up(sample1, sample2):
    '''all the changes will be changed on sample1'''
    r = np.random.beta(32.0, 32.0)
    sample1['img'] = (sample1['img'] * r + sample2['img'] * (1 - r)).astype(np.uint8)
    sample1['anns'] = np.concatenate((sample1['anns'],sample2['anns']), 0)