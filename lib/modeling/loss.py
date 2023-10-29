import torch
import torch.nn.functional as F

from torch import nn
from lib.modeling.matcher import build_matcher
from lib.utils.box_utils import generalized_box_iou, box_cxcywh_to_xyxy
from lib.utils.model_utils import accuracy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, bbox_type, sketch_head):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.bbox_type = bbox_type
        self.sketch_head = sketch_head

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, log=True):
        """
        Classification loss (NLL)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
       
            indices: [(src_idx, tgt_idx), (src_idx, tgt_idx), (src_idx, tgt_idx), ...]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #sentence_slots)
         # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes[idx])[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([num_tgts["num"] for num_tgts in targets["num_targets"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices):
        assert 'pred_boxes' in outputs
        tgts = []
        for tgt_video in targets:  # batch-level
            boxes = []
            for tgt_frame in tgt_video['bboxes'].values():  # video-level
                for tgt_instance in tgt_frame:  # frame-level
                    boxes.append(tgt_instance['bbox'])  # instance-level
            boxes = torch.stack(boxes)
            tgts.append(boxes)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # (#boxes, 2)
        tgt_boxes = torch.cat([t[i] for t, (_, i) in zip(tgts, indices)], dim=0)  # (#boxes, 2)
        tgt_boxes = tgt_boxes.to(src_boxes.device)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.mean()

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  # [0,0,0,1,1,1,1,1,...]
        src_idx = torch.cat([src for (src, _) in indices])  # [0,1,3,6,2,8,3,5,...]
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])  # [0,0,0,1,1,1,1,1,...]
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])  # [1,2,0,2,4,1,0,3,...]
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.sketch_head == 'sketch_detr':
            return self._forward_sketch_detr(outputs, targets)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_bbox_indices, tgt_bbox_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _forward_sketch_detr(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses_list = []
        for output in outputs:
            outputs_without_aux = {k: v for k, v in output.items() if k != 'aux_outputs'}

            # Retrieve the matching between the outputs of the last layer and the targets
            # list(tuples), each tuple is (pred_bbox_indices, tgt_bbox_indices)
            indices = self.matcher(outputs_without_aux, targets)

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, output, targets, indices))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in output:
                for i, aux_outputs in enumerate(output['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        kwargs = {}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            losses_list.append(losses)
        return losses_list


def build_loss(args):
    matcher = build_matcher(args)

    weight_dict = {"loss_bbox": args.set_cost_bbox,
                   "loss_giou": args.set_cost_giou,
                   "loss_label": args.set_cost_class}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    return SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        bbox_type=args.bbox_type,
        sketch_head=args.sketch_head
    )