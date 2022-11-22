import sys
import os
import json
import torch
import numpy as np
from time import time
from pycocotools.cocoeval import COCOeval
from utils.eval_mr import CrowdHumanEval
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
        self.other_forward = args.forward_func

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
            self.model = self.model.to(self.device)
        else:
            print('Please indicating one .pth/.pt file!')
            exit()

    def set_log(self):
        self.log_dir = os.path.dirname(self.args.ckpt_file)
        self.val_img_result_json_name = os.path.splitext(self.args.ckpt_file)[0] + '_evalresult.json'
        self.val_log_name = os.path.splitext(self.args.ckpt_file)[0] + '_fullCOCOresult.log'

    def eval(self, record_name=None):
        self.set_log()
        result_json_found = os.path.exists(self.val_img_result_json_name)
        self.build_eval_loader()
        if not result_json_found: print('Prediction Not Found, Eval the Model')
        if self.args.force_eval or not result_json_found or self.other_forward:
            self.load_model()
            self.model.eval()
            results = []
            print('Begin Infer Val Dataset')
            for i, samples in enumerate(self.loader):
                samples['imgs'] = samples['imgs'].to(self.device).float() / 255
                self.normalizer(samples)
                with torch.no_grad():
                    if self.other_forward:
                        results = self.model.__getattribute__(self.other_forward)(samples)
                    else:
                        results.append(self.model(samples))
                progressbar((i+1)/float(self.itr_in_epoch), barlenth=40)
            self.model.get_stats()
            if self.other_forward:
                exit()
            print('Infer Complete, Sorting')
            self.result_for_json = []
            for result in results:
                self.result_for_json.extend(self.model.coco_parse_result(result))
            with open(self.val_img_result_json_name, 'w') as f:
                json.dump(self.result_for_json, f)
            print('Prediction Result saved in %s'%self.val_img_result_json_name)
        else:
            print('Found Prediction json file %s'%self.val_img_result_json_name)
        val_result = gen_eval(self.val_img_result_json_name, self.loader.dataset.annotations, self.val_log_name,
                              pre_str=record_name, eval_type=self.args.type)
        print('Full COCO result saved in %s'%self.val_log_name)
        return val_result

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

def gen_eval(dt, gt:COCO, log_name, pre_str=None, eval_type='coco'):
    if eval_type == 'coco':
        EVAL_coco = COCOeval
    elif eval_type == 'mr':
        EVAL_coco = CrowdHumanEval
    else:
        raise NotImplementedError('Invalid evaluation type')

    start = time()
    # Print the evaluation result to the log
    ori_std = sys.stdout
    try:
        with open(log_name, "a") as f:
            sys.stdout = f
            if pre_str: print(pre_str)
            dt = gt.loadRes(dt)
            evaluator = EVAL_coco(gt, dt, 'bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            print("eval_times:%.2fs"%(time()-start))
            print("\n")
    except:
        sys.stdout = ori_std
        print('Error in evaluation')
        raise
    sys.stdout = ori_std
    return evaluator.stats[:2]