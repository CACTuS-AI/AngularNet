"""Medformer with GGM layers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BinaryNorm, Conv1dGGM, LinearGGM

PATCH_LENS = (2, 4, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.register_buffer("pe", torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class CrossChannelTokenEmbedding(nn.Module):
    def __init__(self, c_in: int, l_patch: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv2d(1, d_model, kernel_size=(c_in, l_patch), stride=(1, l_patch), bias=False)

    def forward(self, x):
        return self.tokenConv(x)


class ListPatchEmbedding(nn.Module):
    def __init__(self, enc_in: int, d_model: int, patch_lens):
        super().__init__()
        self.paddings = [nn.ReplicationPad1d((0, p)) for p in patch_lens]
        self.value_embeddings = nn.ModuleList(CrossChannelTokenEmbedding(enc_in, p, d_model) for p in patch_lens)
        self.position_embedding = PositionalEmbedding(d_model)
        self.learnable_embeddings = nn.ParameterList(nn.Parameter(torch.zeros(1, d_model)) for _ in patch_lens)

    def forward(self, x):  # [B, L, C]
        x = x.permute(0, 2, 1)
        out = []
        for pad, emb, cxt in zip(self.paddings, self.value_embeddings, self.learnable_embeddings):
            t = emb(pad(x).unsqueeze(1)).squeeze(2).transpose(1, 2)  # [B, patches, d_model]
            out.append(t + cxt + self.position_embedding(t))
        return out


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, scale: float):
        super().__init__()
        self.n_heads, self.scale = n_heads, scale
        self.query_projection = LinearGGM(d_model, d_model)
        self.key_projection = LinearGGM(d_model, d_model)
        self.value_projection = LinearGGM(d_model, d_model)
        self.out_projection = LinearGGM(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.query_projection(x).view(B, L, self.n_heads, -1)
        k = self.key_projection(x).view(B, L, self.n_heads, -1)
        v = self.value_projection(x).view(B, L, self.n_heads, -1)
        A = torch.softmax(self.scale * torch.einsum("blhe,bshe->bhls", q, k), dim=-1)
        out = torch.einsum("bhls,bshd->blhd", A, v).contiguous().view(B, L, -1)
        return self.out_projection(out)


class MedformerLayer(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, n_heads: int):
        super().__init__()
        scale = d_model * math.pi ** 2 / (4 * math.sqrt(d_model // n_heads))
        self.intra_attentions = nn.ModuleList(AttentionLayer(d_model, n_heads, scale) for _ in range(num_blocks))
        self.inter_attention = AttentionLayer(d_model, n_heads, scale) if num_blocks > 1 else None

    def forward(self, xs):
        xs = [layer(x) for x, layer in zip(xs, self.intra_attentions)]
        if self.inter_attention is None:
            return xs
        routers = self.inter_attention(torch.cat([x[:, -1:] for x in xs], dim=1))
        return [torch.cat([x[:, :-1], routers[:, i:i + 1]], dim=1) for i, x in enumerate(xs)]


class EncoderLayer(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = MedformerLayer(num_blocks, d_model, n_heads)
        self.conv1 = Conv1dGGM(d_model, d_ff, kernel_size=1, N_factor=1.0)
        self.conv2 = Conv1dGGM(d_ff, d_model, kernel_size=1, N_factor=1.0)
        self.norm1 = BinaryNorm(d_model)
        self.norm2 = BinaryNorm(d_model)

    def forward(self, xs):
        xs = [self.norm1(x + a) for x, a in zip(xs, self.attention(xs))]
        ys = [self.conv2(F.relu(self.conv1(x.transpose(-1, 1)))).transpose(-1, 1) for x in xs]
        return [self.norm2(x + y) for x, y in zip(xs, ys)]


class Encoder(nn.Module):
    def __init__(self, layers, d_model: int):
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xs):
        for layer in self.attn_layers:
            xs = layer(xs)
        return self.norm(torch.cat(xs, dim=1))


class MedformerGGM(nn.Module):
    """Input [batch, seq_len, enc_in] -> logits [batch, num_class]."""

    def __init__(self, enc_in: int, seq_len: int, num_class: int, d_model: int = 128, n_heads: int = 8,
                 e_layers: int = 6, d_ff: int = 256, patch_lens=PATCH_LENS):
        super().__init__()
        patch_lens = tuple(int(p) for p in patch_lens)
        patch_nums = [int((seq_len - p) / p + 2) for p in patch_lens]
        self.enc_embedding = ListPatchEmbedding(enc_in, d_model, patch_lens)
        self.encoder = Encoder([EncoderLayer(len(patch_lens), d_model, n_heads, d_ff) for _ in range(e_layers)],
                               d_model)
        self.projection = nn.Linear(d_model * sum(patch_nums), num_class)

    def forward(self, x):
        out = F.gelu(self.encoder(self.enc_embedding(x)))
        return self.projection(out.reshape(out.shape[0], -1))
