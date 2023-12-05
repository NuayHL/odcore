import time
import torch
import torch.nn
import numpy as np
import cv2
from copy import deepcopy

from data.data_augment import LetterBox, Normalizer
from data.dataset import CocoDataset
from utils.paralle import de_parallel
from utils.visualization import printImg

class Infer():
    def __init__(self, config, args, model, device):
        self.config = config
        self.args = args
        self.model = model
        self.device = device
        self.letterbox = LetterBox(config.data)
        self.normalizer = Normalizer(config.data, self.device)
        self.load_model()
        self.other_forward = args.forward_func

    def load_model(self):
        print('Model Parameters: ', end='')
        if self.args.ckpt_file != '':
            print(self.args.ckpt_file)
            print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.ckpt_file)
                try:
                    self.model.load_state_dict(ckpt_file['model'])
                except:
                    print('FAIL')
                    print('\t-Parallel Model Loading:', end=' ')
                    self.model.load_state_dict(de_parallel(ckpt_file['model']))
                print('SUCCESS')
            except:
                print("FAIL")
                raise
            model = self.model.to(self.device)
            model.eval()
        else:
            print('Please indicating one .pth/.pt file!')
            exit()

    @torch.no_grad()
    def __call__(self, *imgs, other_forward=None):
        other_forward = other_forward if other_forward else self.other_forward
        ori_imgs = list()
        batched_samples = list()
        for img in imgs:
            if isinstance(img, str):
                img_name = img
                img = cv2.imread(img)
                img = img[:, :, ::-1]
            elif isinstance(img, np.ndarray):
                img_name = 'infer_image'
            else:
                raise NotImplementedError("Unknown inputType")
            ori_imgs.append(deepcopy(img))
            sim_sample = {'img':img, 'anns': np.ones((1, 4)).astype(np.float32),
                          'id': img_name, 'shape': (img.shape[1], img.shape[0])}
            self.letterbox(sim_sample)
            batched_samples.append(sim_sample)
        batched_samples = CocoDataset.OD_default_collater(batched_samples)
        batched_samples['imgs'] = batched_samples['imgs'].to(self.device).float() / 255
        self.normalizer(batched_samples)
        if other_forward:
            results = self.model.__getattribute__(other_forward)(batched_samples)
        else:
            results = self.model(batched_samples)
        self.model.get_stats()
        if other_forward:
            exit()
        return results, ori_imgs

    ## Single Image Infer
    # @torch.no_grad()
    # def __call__(self, img, test_para=None):
    #     print('go to first')
    #     if isinstance(img, str):
    #         img_name = img
    #         img = cv2.imread(img)
    #         img = img[:, :, ::-1]
    #     elif isinstance(img, np.ndarray):
    #         img_name = 'Happy'
    #     else:
    #         raise NotImplementedError("Unknown inputType")
    #     ori_img = deepcopy(img)
    #     sim_sample = {'img':img, 'anns': np.ones((1,4)).astype(np.float32),
    #                   'ids': [img_name], 'shapes': [(img.shape[1],img.shape[0])]}
    #     self.letterbox(sim_sample)
    #     sim_sample['imgs'] = np.transpose(sim_sample['img'], (2, 0, 1))
    #     del sim_sample['img']
    #     sim_sample['imgs'] = torch.from_numpy(sim_sample['imgs']).float() / 255
    #     sim_sample['imgs'] = torch.unsqueeze(sim_sample['imgs'], dim=0)
    #     sim_sample['imgs'] = sim_sample['imgs'].to(self.device)
    #     self.normalizer(sim_sample)
    #     if test_para:
    #         result = self.model.__getattribute__(test_para)(sim_sample)
    #     else:
    #         result = self.model(sim_sample)
    #     return result, ori_img
