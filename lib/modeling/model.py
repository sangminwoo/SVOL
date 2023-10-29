import torch
import torch.nn as nn
from lib.modeling.backbone import build_backbone
from lib.modeling.svanet import build_svanet
# from lib.modeling.svanet_variants import build_svanet
from lib.modeling.sketch_detr import build_sketchdetr


class SketchLocalizationModel(nn.Module):

	def __init__(self, backbone, head):
		super(SketchLocalizationModel, self).__init__()
		self.backbone = backbone
		self.head = head

	def forward(self, src_sketch, src_video, src_sketch_mask=None, src_video_mask=None):
		N, T, C0, H0, W0 = src_video.shape
		src_sketch, src_video = \
			self.backbone(src_sketch, src_video)  # (batch_size, L_sketch), (batch_size, L_vid)

		src_sketch_mask = src_sketch_mask.repeat_interleave(src_sketch.shape[1], dim=1)
		src_video_mask = src_video_mask.repeat_interleave(src_video.shape[1]//T, dim=1)

		outputs = self.head(
			src_sketch, src_sketch_mask,  # (batch_size, L_sketch, D), (batch_size, L_sketch)
			src_video, src_video_mask  # (batch_size, L_vid, D_vid), (batch_size, L_vid)
		)
		return outputs


def build_model(args):
	backbone = build_backbone(args)
	if args.sketch_head == 'svanet':
		head = build_svanet(args)
	elif args.sketch_head == 'sketch_detr':
		head = build_sketchdetr(args)
	else:
		raise NotImplementedError

	model = SketchLocalizationModel(
		backbone,
		head
	)
	return model