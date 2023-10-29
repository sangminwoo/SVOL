# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modeling.transformer import build_transformer
from lib.modeling.cross_modal_transformer import build_cross_modal_transformer
from lib.modeling.position_encoding import build_position_encoding


class SVANet(nn.Module):
    """ End-to-End Sketch-based Video Object Localization with Transformer """

    def __init__(self, transformer, sketch_position_embed, video_position_embed,
                 input_vid_dim, input_skch_dim, num_queries, input_dropout=0.1,
                 aux_loss=True, use_sketch_pos=True, n_input_proj=2, num_classes=2, vis_mode=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            sketch_position_embed: position_embedding for sketch
            video_position_embed: torch module of the position_embedding, See position_encoding.py
            input_vid_dim: int, video feature input dimension
            input_skch_dim: int, sketch feature input dimension
            num_queries: number of queries, ie detection slot.
                         num_queries is the maximal number of objects SVANet can detect in a single video.
            input_dropout: dropout ratio that is applied to the input.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_sketch_pos: whether to enable position_embedding for sketch.
            n_input_proj: number of input projetction layers.
            num_classes: number of classes = 2 (here, we only consider foreground, background).
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.transformer = transformer
        self.sketch_position_embed = sketch_position_embed
        self.video_position_embed = video_position_embed
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.use_sketch_pos = use_sketch_pos
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.n_input_proj = n_input_proj
        self.class_head = nn.Linear(hidden_dim, num_classes)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_video_proj = nn.Sequential(*[
            LinearLayer(input_vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_sketch_proj = nn.Sequential(*[
            LinearLayer(input_skch_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.vis_mode = vis_mode
        self.aux_loss = aux_loss

    def forward(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
        """The forward expects two tensors:
                - src_sketch: [batch_size, L_sketch, D]
                - src_sketch_mask: [batch_size, L_sketch], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
                - src_video: [batch_size, L_video, D_vid]
                - src_video_mask: [batch_size, L_video], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
            It returns a dict with the following elements:
                - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
                - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each indiimgual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_video = self.input_video_proj(src_video)  # (batch_size, L_video, d)
        mask_video = src_video_mask.bool()  # (batch_size, L_video)
        pos_video = self.video_position_embed(src_video, mask_video)  # (batch_size, L_video, d)

        src_sketch = self.input_sketch_proj(src_sketch)  # (batch_size, L_sketch, d)
        mask_sketch = src_sketch_mask.bool()  # (batch_size, L_sketch)
        pos_sketch = self.sketch_position_embed(src_sketch, mask_sketch)  # (batch_size, L_sketch, d)

        hs, att1, att2, att3, att4 = self.transformer(
            src_video, src_sketch, ~mask_video, ~mask_sketch, pos_video, pos_sketch, self.query_embed.weight
        )

        # import matplotlib.pyplot as plt
        # import torch.nn.functional as F
        # # import seaborn as sns

        # # TODO: visualize
        # print('att1', att1)

        # vis_att1 = att1.detach().cpu()[-1][0]  # [#layers, batch_size, L_sketch, L_video] -1: last layer; 0: first in batch
        # vis_att2 = att2.detach().cpu()[-1][0]
        # vis_att3 = att3.detach().cpu()[-1][0]
        # vis_att4 = att4.detach().cpu()[-1][0]

        # vis_att1 = (vis_att1 - vis_att1.min(1)[0]) / vis_att1.max(1)[0] 
        # vis_att2 = (vis_att2 - vis_att2.min(0)[0]) / vis_att2.max(0)[0] 
        # vis_att3 = (vis_att3 - vis_att3.min(0)[0]) / vis_att3.max(0)[0] 
        # vis_att4 = (vis_att4 - vis_att4.min(0)[0]) / vis_att4.max(0)[0]

        # plt.matshow(vis_att1)
        # plt.colorbar()
        # plt.show()
        # plt.matshow(vis_att2)
        # plt.colorbar()
        # plt.show()
        # plt.matshow(vis_att3)
        # plt.colorbar()
        # plt.show()
        # plt.matshow(vis_att4)
        # plt.colorbar()
        # plt.show()

        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, 2)
        outputs_coord = self.bbox_embed(hs)  # (#layers, batch_size, #queries, 4)
        outputs_coord = outputs_coord.sigmoid()
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        if self.vis_mode is not None:
            return out , hs
        else:
            return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_svanet(args):
    transformer = build_cross_modal_transformer(args)
    sketch_position_embed, video_position_embed = build_position_encoding(args)

    return SVANet(
        transformer,
        sketch_position_embed,
        video_position_embed,
        input_vid_dim=args.input_vid_dim,
        input_skch_dim=args.input_skch_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        use_sketch_pos=args.use_sketch_pos,
        n_input_proj=args.n_input_proj,
        vis_mode=args.vis_mode,
    )
