import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CrossModalTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, activation="gelu"):
        super().__init__()

        layer = CrossModalTransformerLayer(d_model, nhead, dim_feedforward, activation)
        self.layers = _get_clones(layer, num_layers)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_skch, vid_mask, skch_mask, vid_pos, skch_pos, query_embed):
        """
        Args:
            src_vid: (batch_size, L, d)
            src_skch: (batch_size, 1, d)
            vid_mask: (batch_size, L)
            skch_mask: (batch_size, L)
            vid_pos: (batch_size, L, d) the same as src_vid
            skch_pos: (batch_size, L_skch, d) the same as src_skch
            query_embed: (#queries, d)

        Returns:

        """
        outputs = []
        att1_weights = []  # sketch-video cross attention
        att2_weights = []  # content self attention
        att3_weights = []  # token self attention
        att4_weights = []  # content-token cross attention

        bs, l, d = src_vid.shape
        src_vid = src_vid.transpose(0, 1)  # (L, batch_size, d)
        src_skch = src_skch.transpose(0, 1)  # (1, batch_size, d)
        vid_pos = vid_pos.transpose(0, 1)  # (L, batch_size, d)
        skch_pos = skch_pos.transpose(0, 1)  # (L_skch, batch_size, d)
        if len(query_embed.shape) != 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)
        
        mem = src_vid
        out = torch.zeros_like(query_embed)  # (#queries, batch_size, d)

        for layer in self.layers:
            mem, out, att1, att2, att3, att4 = layer(
                mem,
                src_skch,
                out,
                vid_mask=vid_mask,
                skch_mask=skch_mask,
                vid_pos=vid_pos,
                skch_pos=skch_pos,
                query_pos=query_embed
            )
            outputs.append(out)
            att1_weights.append(att1)
            att2_weights.append(att2)
            att3_weights.append(att3)
            att4_weights.append(att4)

        outputs = torch.stack(outputs).transpose(1, 2)  # (#layers, batch_size, #queries, d)
        att1_weights = torch.stack(att1_weights)  # (#layers, batch_size, L_vid, L_skch)
        att2_weights = torch.stack(att2_weights)  # (#layers, batch_size, L_vid, L_vid)
        att3_weights = torch.stack(att3_weights)  # (#layers, batch_size, #queries, #queries)
        att4_weights = torch.stack(att4_weights)  # (#layers, batch_size, #queries, L_vid)

        return outputs, att1_weights, att2_weights, att3_weights, att4_weights


class CrossModalTransformerLayer(nn.Module):

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, activation="relu"):
        super().__init__()
        self.sketch_video_cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.content_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp1 = MLP(in_features=d_model, hidden_features=dim_feedforward, activation=activation)
        self.norm3 = nn.LayerNorm(d_model)

        self.token_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm4 = nn.LayerNorm(d_model)
        self.content_token_cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm5 = nn.LayerNorm(d_model)
        self.mlp2 = MLP(in_features=d_model, hidden_features=dim_feedforward, activation=activation)
        self.norm6 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_vid, src_skch, out, vid_mask = None, skch_mask = None,
                vid_pos = None, skch_pos=None, query_pos = None):
        '''
        Args:
            src_vid: [L, bs, d]
            src_skch: [1, bs, d]
            out: [num_queries, bs, d]
            vid_mask: [bs, L]
            vid_pos: [L, bs, d]
            skch_pos: [L_skch, bs, d]
            query_pos: [num_queries, bs, d]

        Returns:
            mem: [L, bs, d]
            out: [num_queries, bs, d]
        '''
        # V1 (1x1)
        q = src_skch
        k = v = self.with_pos_embed(src_vid, vid_pos)
        _, att1 = self.sketch_video_cross_attn(q, k, v)  # att1: [bs, 1, L]
        mem = att1.permute(2, 0, 1) * src_vid  # [L, bs, 1] * [L, bs, d] = [L, bs, d]
        mem = src_vid + mem
        mem = self.norm1(mem)

        # V2 (7x7)
        # q = self.with_pos_embed(src_vid, vid_pos)
        # k = self.with_pos_embed(src_skch, skch_pos)
        # v = src_skch
        # mem, att1 = self.sketch_video_cross_attn(q, k, v)  # att1: [bs, L, L_skch]
        # mem = mem + src_vid
        # mem = self.norm1(mem)

        q = k = self.with_pos_embed(mem, vid_pos)
        v = mem
        mem, att2 = self.content_self_attn(q, k, v)
        mem = mem + v
        mem = self.norm2(mem)
        mem = mem + self.mlp1(mem)
        mem = self.norm3(mem)

        q = k = self.with_pos_embed(out, query_pos)
        v = out
        out, att3 = self.token_self_attn(q, k, v)
        out = out + v
        out = self.norm4(out)

        q = self.with_pos_embed(out, query_pos)
        k = self.with_pos_embed(mem, vid_pos)
        v = mem
        out2, att4 = self.content_token_cross_attn(q, k, v, key_padding_mask=vid_mask)
        out = out + out2
        out = self.norm5(out)
        out = out + self.mlp2(out)
        out = self.norm6(out)

        return mem, out, att1, att2, att3, att4


class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, activation="gelu"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = _get_activation_fn(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_cross_modal_transformer(args):
    return CrossModalTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_layers=args.num_layers,
        dim_feedforward=2048,
    )