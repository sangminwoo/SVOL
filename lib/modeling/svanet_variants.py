# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modeling.transformer import build_transformer
from lib.modeling.position_encoding import build_position_encoding


class SVANet(nn.Module):
    """ End-to-End Sketch-based Video Object Localization with Transformer """

    def __init__(self, transformer, sketch_position_embed, video_position_embed,
                 mode, input_dim, num_queries, input_dropout=0.1, aux_loss=True,
                 use_sketch_pos=True, n_input_proj=2, num_classes=2, vis_mode=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            sketch_position_embed: position_embedding for sketch
            video_position_embed: torch module of the position_embedding, See position_encoding.py
            input_dim: int, imgeo feature input dimension
            num_queries: number of queries, ie detection slot.
                         num_queries is the maximal number of objects SVANet can detect in a single video.
            input_dropout: dropout ratio that is applied to the input.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_sketch_pos: whether to enable position_embedding for sketch.
            n_input_proj: number of input projetction layers.
            num_classes: number of classes = 2 (here, we only consider foreground, background).
        """
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
        self.input_sketch_proj = nn.Sequential(*[
            LinearLayer(input_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_video_proj = nn.Sequential(*[
            LinearLayer(input_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_proj = nn.Sequential(*[
            LinearLayer(input_dim*2, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_query_proj = nn.Sequential(*[
            LinearLayer(input_dim+hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.vis_mode = vis_mode
        self.aux_loss = aux_loss

    def forward(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
        if self.mode == 'concat_to_seq':
            out = self._forward_cat_seq(src_sketch, src_sketch_mask, src_video, src_video_mask)
        elif self.mode == 'append_to_seq':
            out = self._forward_app_seq(src_sketch, src_sketch_mask, src_video, src_video_mask)
        elif self.mode == 'concat_to_qry':
            out = self._forward_cat_qry(src_sketch, src_sketch_mask, src_video, src_video_mask)
        else:
            raise NotImplementedError

        return out

    def _forward_cat_seq(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
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
        src_sketch = src_sketch.repeat(1, src_video.shape[1], 1) # (batch_size, L_video, d)
        src = torch.cat([src_sketch, src_video], dim=2)  # (batch_size, L_video, 2d)
        src = self.input_proj(src)  # (batch_size, L_video, d)
        src_mask = src_video_mask.bool()  # (batch_size, L_video)
        pos = self.video_position_embed(src, src_mask)  # (batch_size, L_video, d)

        hs, memory, att = self.transformer(
            src, ~src_mask, self.query_embed.weight, pos
        )
        # hs: (#layers, batch_size, #queries, d)
        # memory: (batch_size, L_sketch+L_video, d)
        # att: (#layers, batch_size, #queries, L_sketch+L_video)
        sketch_mem = memory[:, :src_sketch.shape[1]]  # (batch_size, L_sketch, d)
        video_mem = memory[:, -src_video.shape[1]:]  # (batch_size, L_video, d)
        sketch_att = [att_[:, :, :src_sketch.shape[1]] for att_ in att]  # (#layers, batch_size, #queries, L_sketch)
        video_att = [att_[:, :, -src_video.shape[1]:] for att_ in att]  # (#layers, batch_size, #queries, L_video)

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

        return out

    def _forward_app_seq(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
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
        src_sketch = self.input_sketch_proj(src_sketch) # (batch_size, L_sketch, d)
        src_video = self.input_video_proj(src_video) # (batch_size, L_video, d)

        pos_sketch = self.sketch_position_embed(src_sketch, src_sketch_mask) if self.use_sketch_pos else torch.zeros_like(src_sketch)  # (batch_size, L_sketch, d)
        pos_video = self.video_position_embed(src_video, src_video_mask)  # (batch_size, L_video, d)

        src = torch.cat([src_sketch, src_video], dim=1)  # (batch_size, L_sketch+L_video, d)
        src_mask = torch.cat([src_sketch_mask, src_video_mask], dim=1).bool()  # (batch_size, L_sketch+L_video)
        pos = torch.cat([pos_sketch, pos_video], dim=1)  # (batch_size, L_sketch+L_video, d)

        hs, memory, att = self.transformer(
            src, ~src_mask, self.query_embed.weight, pos
        )
        # hs: (#layers, batch_size, #queries, d)
        # memory: (batch_size, L_sketch+L_video, d)
        # att: (#layers, batch_size, #queries, L_sketch+L_video)
        sketch_mem = memory[:, :src_sketch.shape[1]]  # (batch_size, L_sketch, d)
        video_mem = memory[:, -src_video.shape[1]:]  # (batch_size, L_video, d)
        sketch_att = [att_[:, :, :src_sketch.shape[1]] for att_ in att]  # (#layers, batch_size, #queries, L_sketch)
        video_att = [att_[:, :, -src_video.shape[1]:] for att_ in att]  # (#layers, batch_size, #queries, L_video)

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

        return out

    def _forward_cat_qry(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
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
        src = self.input_video_proj(src_video)  # (batch_size, L_video, d)
        src_mask = src_video_mask.bool()  # (batch_size, L_video)
        pos = self.video_position_embed(src, src_mask)  # (batch_size, L_video, d)

        bs = src_sketch.shape[0]
        src_sketch = src_sketch.repeat(1, self.query_embed.weight.shape[0], 1).permute(1, 0, 2)  # (#queries, batch_size, D)
        query_weight = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # (#queries, batch_size, d)
        query = torch.cat([query_weight, src_sketch], dim=2)  # (#queries, batch_size, D+d)
        query = self.input_query_proj(query)  # (#queries, batch_size, d)

        hs, memory, att = self.transformer(
            src, ~src_mask, query, pos
        )
        # hs: (#layers, batch_size, #queries, d)
        # memory: (batch_size, L_sketch+L_video, d)
        # att: (#layers, batch_size, #queries, L_sketch+L_video)
        sketch_mem = memory[:, :src_sketch.shape[1]]  # (batch_size, L_sketch, d)
        video_mem = memory[:, -src_video.shape[1]:]  # (batch_size, L_video, d)
        sketch_att = [att_[:, :, :src_sketch.shape[1]] for att_ in att]  # (#layers, batch_size, #queries, L_sketch)
        video_att = [att_[:, :, -src_video.shape[1]:] for att_ in att]  # (#layers, batch_size, #queries, L_video)

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
    transformer = build_transformer(args)
    sketch_position_embed, video_position_embed = build_position_encoding(args)

    return SVANet(
        transformer,
        sketch_position_embed,
        video_position_embed,
        mode=args.mode,
        input_dim=args.feat_dim if args.backbone!='resnet' else 512,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        use_sketch_pos=args.use_sketch_pos,
        n_input_proj=args.n_input_proj,
        vis_mode=args.vis_mode,
    )
