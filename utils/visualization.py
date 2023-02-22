import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import wraps

from utils.misc import progressbar, xywh_x1y1x2y2

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
def show_bbox(img, bboxs=[], type="xywh",color=[0,0,255],score=None, thickness=None, **kwargs):
    print('Received bbox:',len(bboxs))
    img = _add_bbox_img(img, bboxs=bboxs, type=type,color=color,score=score, thickness=thickness, **kwargs)
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
def assign_hot_map(anchors, assignments, shape, img=np.zeros((1,1))*255, gt=np.array([[0,0,0,0]])):
    heatmap = np.zeros((shape[0],shape[1]))
    img = _add_bbox_img(img, bboxs=gt, type='xywh')
    lenth = len(anchors)
    anchors = xywh_x1y1x2y2(anchors)
    for idx,(anchor, assign) in enumerate(zip(anchors, assignments)):
        x1 = int(anchor[0] if anchor[0] > 0 else 0)
        x2 = int(anchor[2] if anchor[2] < shape[1] else shape[1])
        y1 = int(anchor[1] if anchor[1] > 0 else 0)
        y2 = int(anchor[3] if anchor[3] < shape[0] else shape[0])
        if assign == -1:
            heatmap[y1:y2, x1:x2] += 0
        elif assign != 0:
            heatmap[y1:y2, x1:x2] += 1
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

def _add_point_img(img, points, color=[0,0,255], thickness=None):
    img_ = img.copy()
    if thickness == None:
        shape = img.shape[:2]
        minlen = min(shape)
        thickness = int(minlen/600.0)
    for idx, point in enumerate(points):
        point_x = int(point[0])
        point_y = int(point[1])
        img_ = cv2.circle(img_, (point_x, point_y), radius=1, color=color, thickness=thickness)
    return img_

def _add_bbox_img(img, bboxs=[], type="xywh",color=[0,0,255], score=None, thickness=None, **kwargs):
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
        with open("odcore/data/categories_coco.json",'r') as fp:
            classname = json.load(fp)

    if thickness == None:
        shape = img.shape[:2]
        minlen = min(shape)
        thickness = int(minlen/600.0)
    for idx, bbox in enumerate(bboxs):
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        if type == "x1y1x2y2": a, b = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        elif type == "x1y1wh": a, b = (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3])
        else: a, b = (bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)),(bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
        img = np.ascontiguousarray(img)
        img = cv2.rectangle(img,a,b,color, thickness=thickness, **kwargs)
        if score is not None:
            try:
                text = classname[int(score[idx, 1].item())]['name']
            except:
                print("class at %d do not exist" % int(score[idx, 1].item()))
                raise
            text += ":%.2f"%score[idx,0]
            point = list(a)
            point[1] -= 5
            img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_COMPLEX_SMALL, max(1,thickness-2), color=color)
    return img

def _isArrayLike(obj):
    # not working for numpy
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def stack_img(img_list, shape, interval=1):
    '''stack images at shape:(rows, cols), interval filled with 1'''
    rows, cols = shape
    assert len(img_list) == rows * cols, '%d and %d' %(len(img_list), rows * cols)
    assert type(img_list[0]) == np.ndarray
    img_type = img_list[0].dtype
    img_shape = img_list[0].shape[:2]
    img_channel = img_list[0].shape[2]
    insert_col = np.ones((img_shape[0],int(interval),img_channel)).astype(img_type)
    insert_rows = np.ones((1, int((img_shape[1] + int(interval)) * cols - int(interval)), img_channel)).astype(img_type)
    rows_stack_imgs = []
    for row in range(rows):
        if row != 0:
            rows_stack_imgs.append(insert_rows)
        for col in range(cols):
            if col == 0:
                row_img = img_list[row * cols]
                row_img = cv2.resize(row_img, img_shape, interpolation=cv2.INTER_NEAREST)
                if len(row_img.shape) == 2:
                    row_img = np.expand_dims(row_img, 2)
            else:
                add_img = img_list[row * cols + col]
                add_img = cv2.resize(add_img, img_shape, interpolation=cv2.INTER_NEAREST)
                if len(add_img.shape) == 2:
                    add_img = np.expand_dims(add_img, 2)
                row_img = np.concatenate([row_img, insert_col], axis=1)
                row_img = np.concatenate([row_img, add_img], axis=1)
        rows_stack_imgs.append(row_img)
    return np.concatenate(rows_stack_imgs, axis=0)

def generate_hot_bar(max, min, height, width=None):
    height = int(height)
    if width == None:
        width = height / 15
    bar = np.ones((int(height), int(width), 1)).astype(np.float)
    interval = (max-min)/float(height-1)
    for row in range(height):
        r_row = height - row -1
        if r_row == height -1:
            bar[row] *= max
        else:
            bar[row] *= min + r_row * interval
    return bar

class LossLog():
    def __init__(self, file_name, is_main_process=True):
        self.file_name = file_name
        self.is_main_process = is_main_process
        self.read_file()

    def read_file(self):
        with open(self.file_name, "r") as f:
            self.losses = f.readlines()
        self.loss_list = {}
        self.loss_epoch_list = {}
        self.loss_sum_list = []
        self.loss_sum_epoch_list = []
        self.index = []
        self.itr_in_epoch = None
        self.incomplete_last_epoch = False
        self.last_epoch_begin_line = 0
        r_idx = 0
        step_in_epoch = 0
        last_epoch = None
        loss_name_get = False
        total_lenth = len(self.losses)
        for idx, loss in enumerate(self.losses):
            try:
                if not self._is_lossline(loss): continue
                if not loss_name_get:
                    self.loss_name = self._parse_loss_name(loss)
                    self.print("Find loss type: ",self.loss_name)
                    for name in self.loss_name:
                        self.loss_list[name] = []
                        self.loss_epoch_list[name] = []
                    loss_name_get = True

                current_epoch = self._parse_epoch_from_line(loss)
                step_in_epoch += 1

                if last_epoch == None:
                    self.last_epoch_begin_line = idx
                    last_epoch = current_epoch
                    self.loss_sum_epoch_list.append(0)
                    for name in self.loss_epoch_list:
                        self.loss_epoch_list[name].append(0)

                if current_epoch != last_epoch:
                    if self.itr_in_epoch == None:
                        self.itr_in_epoch = step_in_epoch - 1
                    # else:
                    #     assert self.itr_in_epoch == step_in_epoch - 1, 'Invalid Log File'
                    self.last_epoch_begin_line = idx
                    self.loss_sum_epoch_list.append(0)
                    for name in self.loss_epoch_list:
                        self.loss_epoch_list[name].append(0)
                    step_in_epoch = 1
                last_epoch = current_epoch

                self.loss_sum_list.append(0)
                for name in self.loss_name:
                    temp_loss = self._parse_loss_from_line(name, loss)
                    self.loss_list[name].append(temp_loss)
                    self.loss_sum_list[-1] += temp_loss
                    self.loss_epoch_list[name][-1] += temp_loss
                    self.loss_sum_epoch_list[-1] += temp_loss
            except:
                self.print('errorline:', idx)
                raise
            self.index.append(r_idx)
            r_idx += 1
            if self.is_main_process:
                progressbar((idx + 1) / float(total_lenth), barlenth=40)
        last_epoch_steps = step_in_epoch
        self.last_epoch = last_epoch
        if self.itr_in_epoch == None:
            self.print("Warning: The first epoch of the experiments seems not complete! "
                       "Please check if exist last_epoch.pth in experiment files.")
            self.itr_in_epoch = last_epoch_steps
            self.incomplete_last_epoch = False
        else:
            if last_epoch_steps != self.itr_in_epoch:
                self.incomplete_last_epoch = True
                self.print('Incomplete Last Epoch FOUND!')

        for key in self.loss_list:
            self.loss_list[key] = np.array(self.loss_list[key])
        for key in self.loss_epoch_list:
            self.loss_epoch_list[key] = np.array(self.loss_epoch_list[key])
            self.loss_epoch_list[key] /= self.itr_in_epoch
            if self.incomplete_last_epoch:
                self.loss_epoch_list[key][-1] *= self.itr_in_epoch / float(last_epoch_steps)
        self.loss_sum_epoch_list = np.array(self.loss_sum_epoch_list)/self.itr_in_epoch
        if self.incomplete_last_epoch:
            self.loss_sum_epoch_list[-1] *= self.itr_in_epoch / float(last_epoch_steps)
        self.loss_sum_list = np.array(self.loss_sum_list)
        self.index = np.array(self.index)
        self.epoch_index = np.arange(len(self.loss_sum_epoch_list)).astype(np.int32)
        self.total_iter_times = len(self.index)

    def draw_loss(self):
        fig, ax = plt.subplots()
        for name in self.loss_name:
            ax.plot(self.index, self.loss_list[name])
        ax.set(xlabel="Iteration(times)", ylabel="Loss", title="Training Loss for " + self.file_name)
        ax.grid()
        fig.legend(self.loss_name)
        plt.show()

    def draw_sum_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.index, self.loss_sum_list)
        ax.set(xlabel="Iteration(times)", ylabel="Loss", title="Training Loss for " + self.file_name)
        ax.grid()
        fig.legend(['total loss'])
        plt.show()

    def draw_epoch_loss(self):
        fig, ax = plt.subplots()
        for name in self.loss_name:
            ax.plot(self.epoch_index, self.loss_epoch_list[name])
        ax.set(xlabel="Epochs", ylabel="Loss", title="Average Training Loss per Epoch for " + self.file_name)
        ax.grid()
        fig.legend(self.loss_name)
        plt.show()

    def draw_sum_epoch_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.epoch_index, self.loss_sum_epoch_list)
        ax.set(xlabel="Epochs", ylabel="Loss", title="Average Training Loss per Epoch for " + self.file_name)
        ax.grid()
        fig.legend(['total loss'])
        plt.show()

    def drop_incomplete_and_write(self, otherfilename):
        if not self.incomplete_last_epoch: return
        assert otherfilename != self.file_name
        with open(otherfilename,'w') as f:
            f.write(''.join(self.losses))
        with open(self.file_name,'w') as f:
            f.write(''.join(self.losses[:self.last_epoch_begin_line]))

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    @staticmethod
    def _parse_loss_name(string: str):
        loss_name = []
        start = string.find('||')
        string = string[(start + 1):]
        while (':' in string):
            end = string.find(':')
            start = 0
            for i in range(2, 100):
                if string[end - i] == ' ':
                    start = end - i + 1
                    break
            loss_name.append(string[start:end])
            string = string[(end + 1):]
        return loss_name

    @staticmethod
    def _parse_epoch_from_line(lossline: str):
        epoch_s = lossline.find('epoch') + 6
        epoch_e = lossline.find('/')
        return int(lossline[epoch_s:epoch_e])

    @staticmethod
    def _parse_loss_from_line(name:str, lossline: str):
        str_s = lossline.find(name)
        str_s += (len(name) + 1)
        for i in range(15):
            if lossline[str_s + i] != ' ':
                str_s = str_s + i
                break
        for i in range(1, 15):
            if lossline[str_s + i] == ' ':
                str_e = str_s + i
                break
        return float(lossline[str_s:str_e])

    @staticmethod
    def _is_lossline(lossline:str):
        if "||" in lossline: return True
        else: return False

class ValLog:
    coco_metrics = ["IoU", "IoU.5", "IoU.75", "IoUs", "IoUm", "IoUl"]
    mr_metrics = ['AP', 'AR', 'MR']
    crowdhuman_metrics = ['mAP', 'mMR']
    def __init__(self, file_name):
        self.file_name = file_name
        self.real_file_name = os.path.splitext(file_name)[0]
        self.data = dict()
        with open(file_name, "r") as f:
            self.lines = f.readlines()

    def coco_val(self, zero_start=True):
        eval_metrics = self.coco_metrics
        for metric in eval_metrics:
            self.data[metric] = ([], []) if not zero_start else ([0], [0])
        flag = -1
        for id, line in enumerate(self.lines):
            if "Epoch" in line:
                current_epoch = int(line[line.find(':') + 1:])
            if "Acc" in line:
                for metric in eval_metrics:
                    self.data[metric][0].append(current_epoch)
                flag = 0
            elif flag == 0:
                flag += 1
            elif 1 <= flag < 1+len(eval_metrics):
                self.data[eval_metrics[flag-1]][1].append(float(line[-5:]))
                flag += 1
                continue
            else:
                flag = -1

    def mr_val(self, zero_start=True):
        eval_metrics = self.mr_metrics
        for metric in eval_metrics:
            self.data[metric] = ([], []) if not zero_start else ([0], [0])
        flag = -1
        for id, line in enumerate(self.lines):
            if "Epoch" in line:
                current_epoch = int(line[line.find(':') + 1:])
            if "Acc" in line:
                for metric in eval_metrics:
                    self.data[metric][0].append(current_epoch)
                flag = 0
            elif flag == 0:
                flag += 1
            elif 1 <= flag < 1+len(eval_metrics):
                dots = line.find('.')
                self.data[eval_metrics[flag-1]][1].append(float(line[dots-1:]))
                flag += 1
                continue
            else:
                flag = -1

    def mip_val(self, zero_start=True):
        eval_metrics = self.crowdhuman_metrics
        for metric in eval_metrics:
            self.data[metric] = ([], []) if not zero_start else ([0], [0])
        flag = -1
        for id, line in enumerate(self.lines):
            if "Epoch" in line:
                current_epoch = int(line[line.find(':') + 1:])
                for metric in eval_metrics:
                    self.data[metric][0].append(current_epoch)
                flag = 1
            elif 1 <= flag < 1+len(eval_metrics):
                dots = line.find('.')
                self.data[eval_metrics[flag-1]][1].append(float(line[dots-1:]))
                flag += 1
                continue
            else:
                flag = -1

    def coco_draw(self, zero_start=True):
        self.coco_val(zero_start=zero_start)
        fig, ax = plt.subplots()
        for metric in self.coco_metrics:
            ax.plot(*self.data[metric])
        ax.set(xlabel="Epochs", ylabel="AP", title="Evaluation for " + self.real_file_name)
        ax.grid()
        fig.legend(self.coco_metrics)
        plt.show()

    def mr_draw(self, zero_start=True):
        self.mr_val(zero_start=zero_start)
        fig, ax = plt.subplots()
        for metric in self.mr_metrics:
            ax.plot(*self.data[metric])
        ax.set(xlabel="Epochs", ylabel="AP", title="Evaluation for " + self.real_file_name)
        ax.grid()
        fig.legend(self.mr_metrics)
        plt.show()

    def mip_draw(self, zero_start=True):
        self.mip_val(zero_start=zero_start)
        fig, ax = plt.subplots()
        for metric in self.crowdhuman_metrics:
            ax.plot(*self.data[metric])
        ax.set(xlabel="Epochs", ylabel="AP", title="Evaluation for " + self.real_file_name)
        ax.grid()
        fig.legend(self.crowdhuman_metrics)
        plt.show()

import itertools
def draw_matrix(data:np.ndarray, width, height, savefile=None):
    assert data.min() >= 0.0 and data.max() <= 1.0
    plt.figure(figsize=(2*(width+1)/7, 2*height/7))
    plt.imshow(data, cmap=plt.cm.Blues)
    for i, j in itertools.product(range(height), range(width)):
        plt.text(j, i, format(data[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if data[i, j] > 0.5 else "black",
                 fontsize=6)
    plt.axis('off')
    plt.tight_layout()
    if savefile:
        plt.savefig(fname=savefile, dpi=300)
    plt.show()

def draw_scheduler(lf, fin_epoches = 100):
    sim_optimizer = torch.optim.SGD(torch.nn.Conv2d(1,1,1).parameters(),lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(sim_optimizer, lr_lambda=lf)
    index = []
    lrs = []
    for e in range(fin_epoches):
        index.append(e)
        lrs.append(scheduler.get_lr())
        scheduler.step()
    fig, ax = plt.subplots()
    ax.plot(index, lrs)
    ax.grid()
    plt.show()



