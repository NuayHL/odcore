import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import wraps

from utils.misc import progressbar

# decorator for tran img
def tran_img(fun):
    @wraps(fun)
    def finfun(img, *args, **kwargs):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = img[:,:,::-1]
        if isinstance(img, np.ndarray):
            pass
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = img.astype(np.int32)
        temp = fun(img, *args, **kwargs)
        return temp
    return finfun

@tran_img
def printImg(img, title: str='', type = 0):
    if type == 0: plt.imshow(img)
    else: plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def dataset_inspection(dataset, idx, anntype="x1y1wh"):
    sample = dataset[idx]
    img = sample["img"].astype(np.int32)
    anns = sample["anns"]
    show_bbox(img, anns, type=anntype)

def dataset_assign_inspection(dataset, imgid, annsidx=None):
    sample = dataset[imgid]
    img = sample["img"].astype(np.int32)
    anns = np.array(sample["anns"]).astype(np.int32)
    assign_visualization(img, anns, annsidx)

@tran_img
def show_bbox(img, bboxs=[], type="xywh",color=[0,0,255],score=None, **kwargs):
    print('Received bbox:',len(bboxs))
    img = _add_bbox_img(img, bboxs=bboxs, type=type,color=color,score=score, **kwargs)
    printImg(img)

@tran_img
def assign_visualization(img, anns, assignresult, anchors, annsidx=None,
                         anntype="x1y1wh"):
    '''
    :param img:
    :param anns:
    :param annsidx: choose an index refered to certain annotation bbox, default is the middle
    :param anchors: pre defined anchors
    :param assignresult: assign result
    :param anntype: anns bbox type
    :return: img with bbox
    '''
    assert len(assignresult) == len(anchors)
    num_anns = len(anns)
    if annsidx is None:
        annsidx = int(num_anns/2)
    assert annsidx>=0 and annsidx<num_anns, "invalid ann_index for these img, change a suitable \'annsidx\'"

    sp_idx = torch.eq(assignresult, annsidx+1).to("cpu")
    sp_anch = (anchors[sp_idx]).astype(np.int32)
    img = _add_bbox_img(img, sp_anch, type="x1y1x2y2")
    img = _add_bbox_img(img, [anns[annsidx,:]], type=anntype, color=[255,0,0], thickness=3, lineType=8)
    printImg(img)

# hot map of assigned anchors
def assign_hot_map(anchors, assignments, img=np.zeros((1,1))*255, gt=np.array([[0,0,0,0]])):
    heatmap = np.zeros((800,1024))
    img = _add_bbox_img(img, bboxs=gt, type='x1y1wh')
    lenth = len(anchors)
    for idx,(anchor, assign) in enumerate(zip(anchors, assignments)):
        x1 = int(anchor[0] if anchor[0] > 0 else 0)
        x2 = int(anchor[2] if anchor[2] < 1024 else 1024)
        y1 = int(anchor[1] if anchor[1] > 0 else 0)
        y2 = int(anchor[3] if anchor[3] < 800 else 800)
        if assign == -1:
            heatmap[y1:y2, x1:x2] += 1
        elif assign != 0:
            heatmap[y1:y2, x1:x2] += 0
        else:
            heatmap[y1:y2, x1:x2] += 0
        progressbar(float(idx+1)/lenth)

    fig, ax = plt.subplots(1,2)
    im = ax[1].imshow(heatmap)
    cbar = ax[1].figure.colorbar(im, ax=ax)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

def _add_bbox_img(img, bboxs=[], type="xywh",color=[0,0,255], score=None, **kwargs):
    '''
    :param img: str for file path/np.ndarray (w,h,c)
    :param bboxs: one or lists or np.ndarray
    :param type: xywh, x1y1x2y2, x1y1wh
    :param color: red
    :param score: score class, that is n2 np.ndarray
    :param kwargs: related to cv2.rectangle
    :return: img with bbox
    '''
    assert type in ["xywh","x1y1x2y2","x1y1wh"],"the bbox format should be \'xywh\' or \'x1y1x2y2\' or \'x1y1wh\'"
    if isinstance(bboxs, np.ndarray):
        assert len(bboxs.shape)==2 and bboxs.shape[1]>=4, "invalid bboxes shape for np.ndarray"
        bboxs = bboxs.astype(np.int32)
    else:
        bboxs = bboxs if _isArrayLike(bboxs) else [bboxs]
    if score is not None:
        assert len(score)==len(bboxs), "invalid score shape"
        import json
        with open("data/categories_coco.json",'r') as fp:
            classname = json.load(fp)
    for idx, bbox in enumerate(bboxs):
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        if type == "x1y1x2y2": a, b = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        elif type == "x1y1wh": a, b = (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3])
        else: a, b = (bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)),(bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
        img = np.ascontiguousarray(img)
        img = cv2.rectangle(img,a,b,color, **kwargs)
        if score is not None:
            text = classname[int(score[idx,1].item())]['name']
            text += ":%.2f"%score[idx,0]
            point = list(a)
            point[1] += 11
            img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_PLAIN, 1,[255,255,255])
    return img

def _isArrayLike(obj):
    # not working for numpy
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def draw_loss(file_name,outputImgName="loss"):
    with open(file_name,"r") as f:
        losses = f.readlines()
    loss_list = {}
    index = []
    loss_name = []
    r_idx = 0
    loss_name_flag = 0
    total_lenth = len(losses)
    for idx, loss in enumerate(losses):
        try:
            if "||" not in loss:
                continue
            if loss_name_flag == 0:
                loss_name = _parse_loss_name(loss)
                for name in loss_name:
                    loss_list[name] = []
                loss_name_flag = 1
            for name in loss_name:
                str_s = loss.find(name)
                str_s += (len(name) + 1)
                for i in range(6):
                    if loss[str_s+i] != ' ':
                        str_s = str_s + i
                        break
                for i in range(1,6):
                    if loss[str_s+i] == ' ':
                        str_e = str_s+i
                        break
                loss_list[name].append(float(loss[str_s:str_e]))
        except:
            print('errorline:',idx)
            raise
        index.append(r_idx)
        r_idx += 1
        progressbar((idx+1)/float(total_lenth), barlenth=40)
    fig, ax = plt.subplots()
    for name in loss_name:
        ax.plot(index, loss_list[name])
    ax.set(xlabel="Iteration(times)",ylabel="Loss",title="Training Loss for "+file_name)
    ax.grid()
    fig.legend(loss_name)
    savepath = os.path.dirname(file_name)
    fig.savefig(os.path.join(savepath, (outputImgName+".png")),dpi = 300)
    plt.show()

def _parse_loss_name(string: str):
    loss_name = []
    start = string.find('||')
    string = string[(start+1):]
    while (':' in string):
        end = string.find(':')
        start = 0
        for i in range(2,100):
            if string[end-i] == ' ':
                start = end-i+1
                break
        loss_name.append(string[start:end])
        string = string[(end+1):]
    return loss_name


# has a bug
def draw_loss_epoch(file_name, num_in_epoch, outputImgName="loss per epoch", logpath="trainingLog", savepath="trainingLog/lossV"):
    with open(logpath+"/"+file_name,"r") as f:
        losses = f.readlines()
        epoch_loss = 0
        epoch_count = 1
        start_idx = 0
        loss_list = []
        index = []
        for i in losses:
            if "WARNING" in i:
                continue
            if start_idx % num_in_epoch == 0:
                index.append(epoch_count)
                loss_list.append(epoch_loss / num_in_epoch)
                epoch_loss = 0
                epoch_count += 1
            loss = float(i[(i.rfind(":")+1):])
            epoch_loss += loss
            start_idx += 1
        if start_idx % num_in_epoch != 0:
            index.append(epoch_count)
            loss_list.append(epoch_loss / (start_idx % num_in_epoch))
    index = index[1:]
    loss_list = loss_list[1:]
    fig, ax = plt.subplots()
    ax.plot(index, loss_list)
    ax.set(xlabel="Epochs",ylabel="Loss",title="Training Loss per epoch for "+file_name)
    ax.grid()

    fig.savefig(savepath+"/"+file_name+"_"+outputImgName+".png")
    plt.show()

def draw_coco_eval(file_name,save_per_epoch=5, outputImgName="evaluation", logpath="trainingLog", savepath="trainingLog/lossV"):

    index = []
    iou = []
    iou_50 = []
    iou_75 = []
    iou_small = []
    iou_medium = []
    iou_large = []
    start_index = save_per_epoch

    with open(logpath+'/'+file_name,"r") as f:
        lines = f.readlines()
    flag = 0
    for line in lines:
        if "Acc" in line: flag = 1
        elif flag == 1:
            index.append(start_index)
            start_index += save_per_epoch
            flag += 1
        elif flag == 2:
            iou.append(float(line[-5:]))
            flag += 1
        elif flag == 3:
            iou_50.append(float(line[-5:]))
            flag += 1
        elif flag == 4:
            iou_75.append(float(line[-5:]))
            flag += 1
        elif flag == 5:
            iou_small.append(float(line[-5:]))
            flag += 1
        elif flag == 6:
            iou_medium.append(float(line[-5:]))
            flag += 1
        elif flag == 7:
            iou_large.append(float(line[-5:]))
            flag = 0

    fig, ax = plt.subplots()
    ax.plot(index, iou)
    ax.plot(index, iou_50)
    ax.plot(index, iou_75)
    ax.plot(index, iou_small)
    ax.plot(index, iou_medium)
    ax.plot(index, iou_large)
    ax.set(xlabel="Epochs",ylabel="AP",title="Evaluation for "+file_name)
    ax.grid()
    fig.legend(["IoU","IoU.5","IoU.75","IoUs","IoUm","IoUl"])
    fig.savefig(savepath+"/"+file_name+"_"+outputImgName+".png")
    plt.show()


