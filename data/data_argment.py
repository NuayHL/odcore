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
    def __call__(self, sample):
        if np.random.rand() < self.fliplr:
            sample["img"] = sample["img"][:,::-1,:]

            _, width, _ = sample["img"].shape
            sample["anns"][:, 0] = width - sample["anns"][:, 0] - sample["anns"][:, 2]

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