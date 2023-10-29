import json
import time
import copy
import numpy as np
import multiprocessing as mp
from functools import partial
from collections import OrderedDict, defaultdict
from lib.evaluate.utils import compute_average_precision_detection, \
     compute_iou_batch_paired, compute_iou_batch_cross


def compute_average_precision_detection_wrapper(input_triple,
                                                iou_thresholds=np.linspace(0.5, 0.95, 10)):
    video, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, iou_thresholds=iou_thresholds)
    return video, scores


def compute_ap(results, iou_thds=np.linspace(0.5, 0.95, 10),
               num_workers=0, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    preds = defaultdict(list)
    gts = defaultdict(list)

    for res in results:
        video = res["video"]
        sketch = res["sketch"]
        frame = res["frame"]
        pred_boxes = res["pred_boxes"]
        gt_boxes = res["gt_boxes"]

        for pbox in pred_boxes:
            preds[video+sketch].append({
                "frame": frame,
                "top-left-x": pbox[0],
                "top-left-y": pbox[1],
                "bot-right-x": pbox[2],
                "bot-right-y": pbox[3],
                "score": pbox[4]
            })
        for gbox in gt_boxes:
            gts[video+sketch].append({
                "frame": frame,
                "top-left-x": gbox['bbox'][0],
                "top-left-y": gbox['bbox'][1],
                "bot-right-x": gbox['bbox'][2],
                "bot-right-y": gbox['bbox'][3],
            })
    video2ap_list = {}
    data_triples = [[video, gts[video], preds[video]] for video in preds]
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, iou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for video, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                video2ap_list[video] = scores
    else:
        for data_triple in data_triples:
            video, scores = compute_ap_from_triple(data_triple)
            video2ap_list[video] = scores

    ap_array = np.array(list(video2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_recall_at_k(results, iou_thds=np.linspace(0.1, 0.9, 9), k=1):
    # if predicted box has IoU >= iou_thd with GT box, we define it positive
    pred_boxes = [res["pred_boxes"][:k] for res in results]
    gt_boxes = [res["gt_boxes"] for res in results]

    max_ious = []
    for i, (preds, gts) in enumerate(zip(pred_boxes, gt_boxes)):
        gts = [e['bbox'] for e in gts]
        if len(gts) == 0:
            continue
            # max_iou = [0]
        else:
            iou = compute_iou_batch_cross(  # (#preds, #gts)
                np.array(preds),
                np.array(gts)
            )
            max_iou = iou.max(axis=0)  # (#gts, )
        max_ious.extend(max_iou)
    max_ious = np.asarray(max_ious)

    iou_thd2recall_at_k = {}
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    for thd in iou_thds:
        iou_thd2recall_at_k[str(thd)] = \
        float(f"{np.mean(max_ious >= thd) * 100:.2f}")
    miou = float(f"{np.mean(max_ious) * 100:.2f}")
    return iou_thd2recall_at_k, miou


def eval_svol(results, verbose=True, logger=None):
    if verbose:
        start_time = time.time()
    iou_thd2average_precision = compute_ap(results, num_workers=8, chunksize=50)
    iou_thd2recall_at_one, miou_at_one = compute_recall_at_k(results, k=1)
    iou_thd2recall_at_five, miou_at_five = compute_recall_at_k(results, k=5)
    ret_metrics = {
        "SVOL-mAP": iou_thd2average_precision,
        "SVOL-R1": iou_thd2recall_at_one,
        "SVOL-R5": iou_thd2recall_at_five,
        "mIoU@R1": miou_at_one,
        "mIoU@R5": miou_at_five
    }
    if verbose:
        logger.info(f"[eval_svol] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def eval_results(results, verbose=True, logger=None, match_number=False):
    """
    results: list(dict), each dict is {
        'video': 'ILSVRC2015_val_00007040',
        'frame': 0,
        'category': 'airplane',
        'gt_boxes': [
            {'track_id': 0, 'bbox': tensor([0.5246, 0.2771, 0.0336, 0.0819])},
            {'track_id': 1, 'bbox': tensor([0.4926, 0.3757, 0.0336, 0.0764])},
            {'track_id': 2, 'bbox': tensor([0.4641, 0.5118, 0.0375, 0.0764])},
            {'track_id': 3, 'bbox': tensor([0.4336, 0.6500, 0.0344, 0.0722])},
            {'track_id': 4, 'bbox': tensor([0.4156, 0.7771, 0.0344, 0.0736])}],
        'pred_boxes': [
            [0.3758, 0.1655, 0.4336, 0.2822, 0.9966],
            [0.4362, 0.2982, 0.4911, 0.4105, 0.9919],
            [0.3976, 0.3567, 0.4551, 0.4657, 0.9919],
            [0.4091, 0.7127, 0.4840, 0.8326, 0.9899],
            [0.4433, 0.5605, 0.5022, 0.6659, 0.9898],
            [0.4281, 0.4217, 0.4957, 0.5354, 0.9871],
            [0.4863, 0.5458, 0.5460, 0.6532, 0.9865],
            [0.4604, 0.4050, 0.5204, 0.5148, 0.9817],
            [0.4070, 0.6047, 0.4768, 0.7192, 0.9815],
            [0.3898, 0.4756, 0.4465, 0.5864, 0.9763]]
    }
    """
    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    svol_scores = eval_svol(results, verbose=verbose, logger=logger)
    eval_metrics.update(svol_scores)
    svol_scores_brief = {
        # mAP with IoU 0.5/0.75 
        "SVOL-full-mAP": svol_scores["SVOL-mAP"]["average"],
        # recall@1 with IoU 0.1/0.3/0.5/0.7
        "SVOL-full-R1@0.1": svol_scores["SVOL-R1"]["0.1"],
        "SVOL-full-R1@0.3": svol_scores["SVOL-R1"]["0.3"],
        "SVOL-full-R1@0.5": svol_scores["SVOL-R1"]["0.5"],
        "SVOL-full-R1@0.7": svol_scores["SVOL-R1"]["0.7"],
        # recall@5 with IoU 0.1/0.3/0.5/0.7
        "SVOL-full-R5@0.1": svol_scores["SVOL-R5"]["0.1"],
        "SVOL-full-R5@0.3": svol_scores["SVOL-R5"]["0.3"],
        "SVOL-full-R5@0.5": svol_scores["SVOL-R5"]["0.5"],
        "SVOL-full-R5@0.7": svol_scores["SVOL-R5"]["0.7"],
        # mIoU
        "SVOL-full-mIoU@R1": svol_scores["mIoU@R1"],
        "SVOL-full-mIoU@R5": svol_scores["mIoU@R5"]
    }
    eval_metrics_brief.update(
        sorted([(k, v) for k, v in svol_scores_brief.items()], key=lambda x: x[0]))

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics