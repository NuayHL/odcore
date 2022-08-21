import sys
import os
import json
import torch
import numpy as np
from time import time
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from data.dataloader import build_dataloader
from data.data_augment import Normalizer
from utils.misc import progressbar
from utils.paralle import de_parallel

class Eval():
    def __init__(self, config, args, model, device):
        self.config = config
        self.args = args
        self.model = model
        self.device = device
        self.normalizer = Normalizer(config.data, device)

    def build_eval_loader(self):
        self.loader = build_dataloader(self.config.training.val_img_anns_path,
                                  self.config.training.val_img_path,
                                  self.config.data,
                                  self.args.batch_size,
                                  -1, self.args.workers, 'val')
        self.itr_in_epoch = len(self.loader)

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
                    print('\t-Parallel Model Loading:',end=' ')
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

    def set_log(self):
        self.log_dir = os.path.dirname(self.args.ckpt_file)
        self.val_img_result_json_name = self.args.ckpt_file[:-4] + '_evalresult.json'
        self.val_log_name = self.args.ckpt_file[:-4] + '_fullCOCOresult.log'

    def eval(self, force_eval=True):
        self.set_log()
        result_json_found = os.path.exists(self.val_img_result_json_name)
        self.build_eval_loader()
        if not result_json_found: print('Prediction Not Found, Eval the Model')
        if force_eval or not result_json_found:
            self.load_model()
            self.model.eval()
            results = []
            print('Begin Infer Val Dataset')
            for i, samples in enumerate(self.loader):
                samples['imgs'] = samples['imgs'].to(self.device).float() / 255
                self.normalizer(samples)
                with torch.no_grad():
                    results.append(self.model(samples))
                progressbar((i+1)/float(self.itr_in_epoch), barlenth=40)
            print('Infer Complete, Sorting')
            self.result_for_json = []
            for result in results:
                self.result_for_json.extend(self.model.coco_parse_result(result))
            with open(self.val_img_result_json_name, 'w') as f:
                json.dump(self.result_for_json, f)
            print('Prediction Result saved in %s'%self.val_img_result_json_name)
        else:
            print('Found Prediction json file %s'%self.val_img_result_json_name)
        coco_eval(self.val_img_result_json_name, self.loader.dataset.annotations, self.val_log_name)
        print('Full COCO result saved in %s'%self.val_log_name)

def coco_eval(dt, gt:COCO, log_name, pre_str=None):
    '''return map, map50'''
    start = time()
    # Print the evaluation result to the log
    ori_std = sys.stdout
    try:
        with open(log_name,"a") as f:
            sys.stdout = f
            if pre_str: print(pre_str)
            dt = gt.loadRes(dt)
            eval = COCOeval(gt, dt, 'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            print("eval_times:%.2fs"%(time()-start))
            print("\n")
    except:
        sys.stdout = ori_std
        print('Error in evaluation')
        raise
        #return 0.0, 0.0
    sys.stdout = ori_std
    return eval.stats[:2]

def model_inference_coconp(loader, model, config):
    """
    return a result np.ndarray for COCOeval
    formate: imgidx x1y1wh score class
    """
    model.eval()
    result_list = []
    result_np = np.ones((0,7), dtype=np.float32)
    lenth = len(loader)
    print('Starting inference.')
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            imgs = batch['imgs'].to(device)
        else:
            imgs = batch['imgs']
        result_list_batch = model(imgs)
        result_list += result_list_batch
        progressbar(float((idx+1)/lenth),barlenth=40)
    print('Sorting...')
    lenth = len(result_list)
    for idx, result in enumerate(result_list):
        if result is not None:
            img_id = dataset.image_id[idx]
            ori_img = dataset.annotations.loadImgs(img_id)[0]
            fx = ori_img['width']/config.input_width
            fy = ori_img['height']/config.input_height
            result_formated = result.to_evaluation(img_id)
            result_formated[:, 1] *= fx
            result_formated[:, 3] *= fx
            result_formated[:, 2] *= fy
            result_formated[:, 4] *= fy
            result_np = np.concatenate((result_np,result_formated),0)
        else:
            pass
        progressbar(float((idx + 1) / lenth), barlenth=40)
    return result_np