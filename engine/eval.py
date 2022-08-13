import sys
import torch
import numpy as np
from time import time
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from utils.misc import progressbar

def coco_eval(dt, gt:COCO, config, log_name):
    start = time()
    # Print the evaluation result to the log
    ori_std = sys.stdout
    with open(log_name+".txt","a") as f:
        sys.stdout = f
        dt = gt.loadRes(dt)
        eval = COCOeval(gt, dt, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        print("eval_times:%.2fs"%(time()-start))
        print("\n")
    sys.stdout = ori_std

def model_inference_coconp(val_loader, model, config):
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