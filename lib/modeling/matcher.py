# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from lib.utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou


class PerFrameMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 num_frames: int = 32, num_queries_per_frame: int = 10):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bbox in the matching cost
            num_frames: sampled frames from video
            num_queries_per_frame: number of query slots per frame
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_queries_per_frame = num_queries_per_frame
        self.foreground_label = 0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted bbox coordinates
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "bboxes": Tensor of dim [num_target_bboxes, 4] containing the target bbox coordinates.
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_bboxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        assert num_queries == self.num_frames * self.num_queries_per_frame

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * #queries, 2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * #queries, 4]

        tgt_bbox = []
        num_boxes = []
        for tgt_video in targets:  # batch-level
            num_boxes.extend(tgt_video['num_boxes_per_frame'])
            for tgt_frame in tgt_video['bboxes'].values():  # video-level
                for tgt_instance in tgt_frame:  # frame-level
                    tgt_bbox.append(tgt_instance['bbox'])  # instance-level

        tgt_bbox = torch.stack(tgt_bbox).to(out_bbox.device)  # [total #boxes in batch, 4]
        tgt_ids = torch.full([tgt_bbox.shape[0]], self.foreground_label)   # [total #boxes in batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * #queries, total #boxes in batch]

        # Compute the L1 cost between bboxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [batch_size * #queries, total #boxes in batch]

        # Compute the giou cost between bboxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  # [batch_size * #queries, total #boxes in batch]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class  # [batch_size * #queries, total #boxes in batch]
        C = C.view(bs*self.num_frames, self.num_queries_per_frame, -1).cpu()  # [batch_size * #frames, #queries in frame, total #boxes in batch]

        # Get pairs of (pred, target)
        cum_num_boxes = np.cumsum(num_boxes)
        tgt_offsets = [0] + list(cum_num_boxes[:-1])
        indices = []
        for i, (c, o) in enumerate(zip(C.split(num_boxes, -1), tgt_offsets)):
            pred_ind, tgt_ind = linear_sum_assignment(c[i])
            pred_ind += i * self.num_queries_per_frame
            tgt_ind += o
            indices.append((pred_ind, tgt_ind))

        # yield successive #frames-sized chunks from lists
        indices = [indices[i:i+self.num_frames] for i in range(0, len(indices), self.num_frames)]  # [batch_size, #frames]

        # aggregate frame-level indices into video-level indices
        # and adjust prediction & target indices
        cum_num_queries = np.cumsum([num_queries] * bs)
        pred_offsets = [0] + list(cum_num_queries[:-1])
        indices_per_video = []
        for indices_, pred_offset in zip(indices, pred_offsets):
            pred_indices_per_video = []
            tgt_indices_per_video = []
            for indice in indices_:
                pred_ind, tgt_ind = indice
                for pred_idx, tgt_idx in zip(pred_ind, tgt_ind):
                    pred_indices_per_video.append(pred_idx - pred_offset)
                    tgt_indices_per_video.append(tgt_idx)
            tgt_indices_per_video = np.asarray(tgt_indices_per_video)
            tgt_indices_per_video = list(tgt_indices_per_video - np.min(tgt_indices_per_video))
            indices_per_video.append((pred_indices_per_video, tgt_indices_per_video))
        indices = indices_per_video

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.foreground_label = 0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * #queries, 2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * #queries, 4]

        tgt_bbox = []
        sizes = []
        for tgt_video in targets:  # batch-level
            count = 0
            for tgt_frame in tgt_video['bboxes'].values():  # video-level
                count += len(tgt_frame)
                for tgt_instance in tgt_frame:  # frame-level
                    tgt_bbox.append(tgt_instance['bbox'])  # instance-level
            sizes.append(count)

        tgt_bbox = torch.stack(tgt_bbox).to(out_bbox.device)  # [total #boxes in batch, 4]
        tgt_ids = torch.full([tgt_bbox.shape[0]], self.foreground_label)   # [total #boxes in batch]

        cost_class = -out_prob[:, tgt_ids]  # [batch_size * #queries, total #boxes in batch]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [batch_size * #queries, total #boxes in batch]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  # [batch_size * #queries, total #boxes in batch]

        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class  # [batch_size * #queries, total #boxes in batch]
        C = C.view(bs, num_queries, -1).cpu()  # [batch_size, #queries, total #boxes in batch]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]  # [batch_size, #frames]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.matcher == 'per_frame_matcher':
        return PerFrameMatcher(
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            cost_class=args.set_cost_class,
            num_frames=args.num_frames,
            num_queries_per_frame=args.num_queries_per_frame
        )
    elif args.matcher == 'video_matcher':
        return HungarianMatcher(
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            cost_class=args.set_cost_class,
        )
    else:
        raise NotImplementedError