"""
Modified from MMAction2
https://github.com/open-mmlab/mmaction2/blob/master/mmaction/core/evaluation/eval_detection.py
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_curve


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def box_area(corners):
    """
    Calculate the area of a box given the corners:

    Args:
      corners: float array of shape (N, 4)
        with the values [x1, y1, x2, y2] for
        each batch element.

    Returns:
      area: (N, 1) tensor of box areas for
        all boxes in the batch
    """
    x1 = corners[..., 0]
    y1 = corners[..., 1]
    x2 = corners[..., 2]
    y2 = corners[..., 3]
    return (x2 - x1) * (y2 - y1)


def compute_iou_batch_paired(box1, box2):
    """
    Calculate the intersection over union of bounding boxes
    in a pair-wise manner.

    Args:
      box1, box2: arrays of shape (N, 4)
        with the values [x1, y1, x2, y2] for
        each batch element.
    Returns:
      iou: array of shape (N, 1) giving
        the intersection over union of boxes between
        box1 and box2.   
    """
    xmin = np.maximum(box1[..., 0], box2[..., 0])  # (N, )
    ymin = np.maximum(box1[..., 1], box2[..., 1])  # (N, )
    xmax = np.minimum(box1[..., 2], box2[..., 2])  # (N, )
    ymax = np.minimum(box1[..., 3], box2[..., 3])  # (N, )
    
    intersection_box = np.stack([xmin, ymin, xmax, ymax], axis=-1)  # (N, 4)

    intersection_area = box_area(intersection_box)
    box1_area = box_area(box1)
    box2_area = box_area(box2)

    union_area = (box1_area + box2_area) - intersection_area

    # If x1 is greater than x2 or y1 is greater than y2
    # then there is no overlap in the bounding boxes.
    # Find the indices where there is a valid overlap.
    valid = np.logical_and(xmin <= xmax, ymin <= ymax)

    # For the valid overlapping boxes, calculate the intersection
    # over union. For the invalid overlaps, set the value to 0.  
    iou = np.where(valid, (intersection_area / union_area), 0)

    return iou


def compute_iou_batch_cross(box1, box2):
    """
    Calculate the intersection over union across the
    all possible pairs of bounding boxes.

    Args:
      box1, box2: arrays of shape (N, 4), (M, 4)
        with the values [x1, y1, x2, y2] for
        each batch element.
    Returns:
      iou: array of shape (N, M) giving
        the intersection over union of boxes between
        box1 and box2.   
    """
    N = box1.shape[0]
    M = box2.shape[0]
    box1 = np.tile(box1, (M,1))  # (N*M, 4) same as torch repeat
    box2 = np.repeat(box2, N, axis=0)  # (M*N, 4) same as torch repeat_interleave 
    
    iou = compute_iou_batch_paired(box1, box2)  # (N*M, )
    iou = iou.reshape(N, M)  # (N, M)

    return iou


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted boxes, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'frame', 'top-left-x', 
            'top-left-y', 'bot-right-x', and 'bot-right-y'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'frame', 'top-left-x',
            'top-left-y', 'bot-right-x', and 'bot-right-y'
            and 'score'.
        iou_thresholds (np.ndarray): A 1d array indicates the intersection
            over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(iou_thresholds)  # K
    num_gts = len(ground_truth)  # N
    num_preds = len(prediction)  # M
    ap = np.zeros(num_thresholds)  # K
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)  # N
    lock_gt = np.ones((num_thresholds, num_gts)) * -1  # (K, N)
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])  # (M, 4)
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))  # (K, M)
    fp = np.zeros((num_thresholds, num_preds))  # (K, M)

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['frame'], []).append(item)

    # Assigning true positive to truly ground truth instances.
    for idx, pred in enumerate(prediction):
        if pred['frame'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['frame']]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred['top-left-x'], pred['top-left-y'],
                           pred['bot-right-x'], pred['bot-right-y']]])
        _gt = np.array([[gt['top-left-x'], gt['top-left-y'],
                         gt['bot-right-x'], gt['bot-right-y']] for gt in gts])
        iou_arr = compute_iou_batch_cross(_pred, _gt)
        iou_arr = iou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest iou score.
        iou_sorted_idx = iou_arr.argsort()[::-1]
        for t_idx, iou_threshold in enumerate(iou_thresholds):
            for j_idx in iou_sorted_idx:
                if iou_arr[j_idx] < iou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(iou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])
    return ap


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision

    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    """
    # Check inputs
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # True labels are all zeros
            # raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])