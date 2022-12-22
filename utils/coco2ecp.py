import os

import json
from collections import defaultdict

def _default_ecp_det_dict():
    return {"identity": 'frame', "children": []}

def coco2ecp_det(coco_det_path):
    """
    ecp_det format:
    {
    "identity": 'frame'
    "children": [{"x0": ,"x1": ,"y0": ,"y1": ,"score": ,
                  "identity": 'pedestrian',
                  "orient": 0.0}]
    }
    """
    import glob
    eval_id = glob.glob('ECP/%s/labels/%s/*/*' % ('day', 'val'))
    eval_id = [os.path.splitext(os.path.basename(img_path))[0] for img_path in eval_id]

    storage_path = os.path.splitext(coco_det_path)[0]
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    ecp_det = defaultdict(_default_ecp_det_dict)
    with open(coco_det_path) as f:
        coco_det = json.load(f)
    print('Convert COCO style to ECP eval type:')
    bar_convert = ProgressBar(coco_det)
    for det_box in coco_det:
        img_id = det_box['image_id']
        box = det_box['bbox']
        score = det_box['score']
        ecp_det[img_id]['children'].append({"x0": box[0], "x1": box[0]+box[2],
                                            "y0": box[1], "y1": box[1]+box[3],
                                            "score": score, "identity": 'pedestrian',
                                            "orient": 0.0})
        bar_convert.update()
    print('Dump to json file')
    bar_dump = ProgressBar(eval_id)
    for image_det in eval_id:
        det_file = image_det + '.json'
        json.dump(ecp_det[image_det], open(os.path.join(storage_path, det_file), 'w'), indent=1)
        bar_dump.update()

    return storage_path

class ProgressBar:
    def __init__(self, iters, barlenth=20, endstr=''):
        self._count = 0
        self._all = 0
        self._len = barlenth
        self._end = endstr
        if isinstance(iters, int):
            self._all = iters
        elif hasattr(iters, '__len__'):
            self._all = len(iters)
        else:
            raise NotImplementedError

    def update(self, step: int = 1, endstr=''):
        self._count += step
        if self._count == self._all:
            endstr += '\n'
        percentage = float(self._count) / self._all
        print('\r[' + '>' * int(percentage * self._len) +
              '-' * (self._len - int(percentage * self._len)) + ']',
              format(percentage * 100, '.1f'), '%',
              end=' ' + self._end + ' ' + endstr)


if __name__ == "__main__":
    path = os.getcwd()
    os.chdir(os.path.join(path, '../../'))
    coco2ecp_det('running_log/YOLOX_ori_ECP_1/best_epoch_evalresult.json')
