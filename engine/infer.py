import time
import torch
import torch.nn
import numpy as np
import cv2
from copy import deepcopy

from data.data_augment import LetterBox, Normalizer

class Infer():
    def __init__(self, config, args, model, device):
        self.config = config
        self.args = args
        self.model = model
        self.device = device
        self.letterbox = LetterBox(config.data)
        self.normalizer = Normalizer(config.data,self.device)
        self.load_model()

    def load_model(self):
        print('FineTuning Model: ', end='')
        if self.args.ckpt_file != '':
            print(self.args.ckpt_file)
            print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                self.model.load_state_dict(ckpt_file['model'])
                print('SUCCESS')
            except:
                print("FAIL")
                raise
            model = self.model.to(self.device)
            model.eval()
        else:
            print('Please indicating one .pth/.pt file!')
            exit()

    def __call__(self, img):
        if isinstance(img, str):
            img_name = img
            img = cv2.imread(img)
            img = img[:, :, ::-1]
        elif isinstance(img, np.ndarray):
            img_name = 'Happy'
        else:
            raise NotImplementedError("Unknown inputType")
        ori_img = deepcopy(img)
        sim_sample = {'img':img, 'anns': np.ones((1,4)).astype(np.float32),
                      'ids': [img_name], 'shapes': [(img.shape[1],img.shape[0])]}
        self.letterbox(sim_sample)
        sim_sample['imgs'] = np.transpose(sim_sample['img'], (2, 0, 1))
        del sim_sample['img']
        sim_sample['imgs'] = torch.from_numpy(sim_sample['imgs']).float() / 255
        sim_sample['imgs'] = torch.unsqueeze(sim_sample['imgs'], dim=0)
        sim_sample['imgs'] = sim_sample['imgs'].to(self.device)
        self.normalizer(sim_sample)
        result = self.model(sim_sample)
        return result, ori_img
