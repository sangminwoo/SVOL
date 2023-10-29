# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modeling.transformer import build_transformer
from lib.modeling.position_encoding import build_position_encoding


class SketchDETR(nn.Module):
    """ Reimplementation of Localizing infinite-shaped fishes: Sketch-guided object localization in the wild """
    def __init__(self, transformer, sketch_position_embed, video_position_embed,
                 mode, input_dim, num_queries, input_dropout=0.1, aux_loss=True,
                 use_sketch_pos=True, n_input_proj=2, num_classes=2):
        super().__init__()
        self.mode = mode
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
            LinearLayer(input_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_query_proj = nn.Sequential(*[
            LinearLayer(input_dim+hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.aux_loss = aux_loss

    def forward(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
        bs, len_vid, _ = src_video.shape
        outputs = []
        for i in range(len_vid):
            src = self.input_video_proj(src_video[:, i, :].unsqueeze(1))  # (batch_size, 1, d)
            src_mask = src_video_mask[:, i].unsqueeze(1).bool()  # (batch_size, 1)
            pos = self.video_position_embed(src, src_mask)  # (batch_size, 1, d)

            src_sketch_ = src_sketch.repeat(1, self.query_embed.weight.shape[0], 1).permute(1, 0, 2)  # (#queries, batch_size, D)
            query_weight = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)
            query = torch.cat([query_weight, src_sketch_], dim=2)  # (#queries, batch_size, D+d)
            query = self.input_query_proj(query)  # (#queries, batch_size, d)

            hs, _, _ = self.transformer(src, ~src_mask, query, pos)
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
            outputs.append(out)
        return outputs


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


def build_sketchdetr(args):
    transformer = build_transformer(args)
    sketch_position_embed, video_position_embed = build_position_encoding(args)

    return SketchDETR(
        transformer,
        sketch_position_embed,
        video_position_embed,
        mode=args.mode,
        input_dim=args.feat_dim,
        num_queries=100,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        use_sketch_pos=args.use_sketch_pos,
        n_input_proj=args.n_input_proj,
    )