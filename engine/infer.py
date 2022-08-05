import torch
import torch.nn
import numpy as np
import cv2

from data.data_augment import LetterBox, Resizer

class Infer():
    def __init__(self, config, args, model, device):
        self.config = config
        self.args = args
        self.model = model
        self.device = device
        self.letterbox = LetterBox(config.data)
        self.load_model()

    def load_model(self):
        print('FineTuning Model: ', end='')
        if self.args.fine_tune != '':
            print(self.args.fine_tune)
            print('\t-Loading:', end=' ')
            try:
                ckpt_file = torch.load(self.args.fine_tune)
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
        sim_sample = {'img':img, 'anns': np.ones((1,4)).astype(np.float32),
                      'id': img_name, 'shape:': (img.shape[1],img.shape[0])}
        self.letterbox(sim_sample)
        sim_sample['img'] = np.transpose(sim_sample['img'], (2, 0, 1))
        sim_sample['img'] = torch.from_numpy(sim_sample['img']).float() / 255
        sim_sample['img'] = torch.unsqueeze(sim_sample['img'], dim=0)
        sim_sample['img'] = sim_sample['img'].to(self.device)
        result = self.model(sim_sample)
        return result[0].to_ori_label((self.config.data.input_width,
                                      self.config.data.input_height))
