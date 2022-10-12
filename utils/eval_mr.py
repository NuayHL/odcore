# https://github.com/TencentYoutuResearch/PedestrianDetection-NohNMS/blob/main/detectron2/evaluation/crowdhuman_evaluation.py
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved
"""
@contact:
penghaozhou@tencent.com
"""
import time
import copy
import datetime
import numpy as np

from collections import defaultdict

class CrowdHumanEval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        if not iouType:
            print("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.params = {}  # evaluation parameters
        self.evalImgs = defaultdict(
            list
        )  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = CrowdHumanParams(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.scorelist = []
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self, id_setup):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        self._ignNum = 0
        self._gtNum = 0
        self._imageNum = len(self.cocoGt.getImgIds())
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = (
                1
                if (
                    gt["height"] < self.params.HtRng[id_setup][0]
                    or gt["height"] > self.params.HtRng[id_setup][1]
                )
                or (
                    gt["vis_ratio"] < self.params.VisRng[id_setup][0]
                    or gt["vis_ratio"] > self.params.VisRng[id_setup][1]
                )
                else gt["ignore"]
            )
            if gt["ignore"] == 1:
                self._ignNum += 1
            else:
                self._gtNum += 1

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self, id_setup):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        VisRng = self.params.VisRng[id_setup]
        self.evalImgs = [
            evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
            for catId in catIds
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, gt, dt):
        p = self.params
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["ignore"]) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, pyiscrowd):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2, gx2) - max(dx1, gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2, gy2) - max(dy1, gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t) / unionarea
        return ious

    def evaluateImg(self, imgId, catId, hRng, vRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"]:
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        # exclude dt out of height range
        dt = [
            d
            for d in dt
            if d["height"] >= hRng[0] / self.params.expFilter
            and d["height"] < hRng[1] * self.params.expFilter
        ]
        dtind = np.array([int(d["id"] - dt[0]["id"]) for d in dt])

        # load computed ious
        if len(dtind) > 0:
            ious = self.computeIoU(gt, dt)
        else:
            ious = []

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    bstOa = iou
                    bstg = -2
                    bstm = -2
                    for gind, g in enumerate(gt):
                        m = gtm[tind, gind]
                        # if this gt already matched, and not a crowd, continue
                        if m > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if bstm != -2 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < bstOa:
                            continue
                        # if match successful and best so far, store appropriately
                        bstOa = ious[dind, gind]
                        bstg = gind
                        if gtIg[gind] == 0:
                            bstm = 1
                        else:
                            bstm = -1

                    # if match made store id of match for both dt and gt
                    if bstg == -2:
                        self.scorelist.append((0, d["score"]))
                        continue
                    dtIg[tind, dind] = gtIg[bstg]
                    dtm[tind, dind] = gt[bstg]["id"]
                    if bstm == 1:
                        gtm[tind, bstg] = d["id"]
                        self.scorelist.append((1, d["score"]))
                    elif bstm != -1:
                        self.scorelist.append((0, d["score"]))

        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "hRng": hRng,
            "vRng": vRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))  # -1 for the precision of absent categories

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1]  # _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind="mergesort")

                dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e["gtIgnore"] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp) / I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side="right") - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except:
                        pass
                    ys[t, :, k, m] = np.array(q)
        self.eval = {
            "params": p,
            "counts": [T, R, K, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TP": ys,
        }
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self, id_setup=0):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(iouThr=None, maxDets=100):
            p = self.params
            iStr = " {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}% \n"
            titleStr = "Average Miss Rate"
            typeStr = "(MR)"
            setupStr = p.SetupLbl[id_setup]
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )
            heightStr = "[{:0.0f}:{:0.0f}]".format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = "[{:0.2f}:{:0.2f}]".format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval["TP"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1 - s[:, :, :, mind]

            if len(mrs[mrs < 2]) == 0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs < 2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)
            return mean_s

        def eval_AP():
            def _calculate_map(recall, precision):
                assert len(recall) == len(precision)
                area = 0
                for i in range(1, len(recall)):
                    delta_h = (precision[i - 1] + precision[i]) / 2
                    delta_w = recall[i] - recall[i - 1]
                    area += delta_w * delta_h
                return area

            self.scorelist.sort(key=lambda x: x[1], reverse=True)
            tp, fp = 0.0, 0.0
            rpX, rpY = list(), list()
            total_det = len(self.scorelist)
            total_gt = self._gtNum
            total_images = self._imageNum
            fpn = []
            recalln = []
            thr = []
            fppi = []
            for i, item in enumerate(self.scorelist):
                if item[0] == 1:
                    tp += 1.0
                elif item[0] == 0:
                    fp += 1.0
                fn = total_gt - tp
                recall = tp / (tp + fn)
                precision = tp / (tp + fp + 1e-3)
                rpX.append(recall)
                rpY.append(precision)
                fpn.append(fp)
                recalln.append(tp)
                fppi.append(fp / total_images)

            AP = _calculate_map(rpX, rpY)
            return AP, rpX[-1]

        ap, recall = eval_AP()
        mr = _summarize(iouThr=0.5, maxDets=1000)
        print("--------------------------------------")
        print("|AP: {:.4} | Recall:{:.4}|  MR:{:.4}|".format(ap, recall, mr))
        print("--------------------------------------")
        if not self.eval:
            raise Exception("Please run accumulate() first")
        return {"AP": ap, "Recall": recall, "MR": mr}

    def __str__(self):
        self.summarize()


class CrowdHumanParams:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value

        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.fppiThrs = np.array(
            [0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000]
        )
        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        self.iouThrs = np.array([0.5])

        self.HtRng = [[0, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2]]
        self.VisRng = [[0.0, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2, 0.65]]
        self.SetupLbl = ["MR", "Reasonable"]

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None