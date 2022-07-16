import numpy as np
import cv2

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
        self.hsv_h = config_data.hsv_h
        self.hsv_s = config_data.hsv_s
        self.hsv_v = config_data.hsv_v
    def __call__(self, sample):
        if np.random.rand() < self.fliplr:
            sample["img"] = sample["img"][:,::-1,:]

            _, width, _ = sample["img"].shape
            sample["anns"][:, 0] = width - sample["anns"][:, 0] - sample["anns"][:, 2]
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

class Resizer():
    def __init__(self, config_data):
        self.width = config_data.input_width
        self.height = config_data.input_height
    def __call__(self, sample, w0, h0):
        fy = self.height / float(h0)
        fx = self.width / float(w0)
        sample["anns"][:, 0] *= fx
        sample["anns"][:, 2] *= fx
        sample["anns"][:, 1] *= fy
        sample["anns"][:, 3] *= fy
        sample["img"] = cv2.resize(sample["img"], (self.width, self.height))