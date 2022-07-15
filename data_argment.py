import numpy as np
import cv2

class Normalizer():
    def __init__(self):
        self.mean = np.array([0.46431773, 0.44211456, 0.4223358])
        self.std = np.array([0.29044453, 0.28503336, 0.29363019])
    def __call__(self, sample):
        img = sample['img']
        img = img.astype(np.float32)/255
        img = (img - self.mean)/self.std
        return {'img':img, 'anns':sample['anns']}

class Augmenter():
    def __call__(self, sample, filp_x=0.5):
        if np.random.rand() < filp_x:
            img, anns = sample["img"], sample["anns"]
            img = img[:,::-1,:]

            _, width, _ = img.shape
            anns[:, 0] = width - anns[:, 0] - anns[:, 2]

            sample = {'img':img, 'anns':anns}


        return sample

class Resizer():
    def __init__(self, input_height, input_width):
        self.width = input_width
        self.height = input_height
    def __call__(self, sample):
        img, anns = sample["img"], sample["anns"].astype(np.float32)
        fy = self.height / float(img.shape[0])
        fx = self.width / float(img.shape[1])
        anns[:, 0] = fx * anns[:, 0]
        anns[:, 2] = fx * anns[:, 2]
        anns[:, 1] = fy * anns[:, 1]
        anns[:, 3] = fy * anns[:, 3]
        img = cv2.resize(img, (self.width, self.height))
        return {'img':img, 'anns':anns}