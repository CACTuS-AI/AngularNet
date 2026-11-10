"""BERT-base with GGM layers (BiBERT setup) for GLUE sentence classification."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BinaryNorm, PackedLinearGGM


class TernaryEmbedding(nn.Module):
    """Word embeddings stored as 2-bit codes (0 -> -1, 1 -> 0, 2 -> +1) and one scale per row."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("packed_codes", torch.zeros(num_embeddings, (self.dim + 3) // 4, dtype=torch.uint8))
        self.register_buffer("row_scale", torch.ones(num_embeddings))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        codes = self.packed_codes[ids]
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=codes.device)
        symbols = ((codes.unsqueeze(-1) >> shifts) & 3).reshape(*codes.shape[:-1], -1)[..., :self.dim]
        return symbols.to(torch.float32).sub_(1) * self.row_scale[ids].unsqueeze(-1)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, layer_norm_eps):
        super().__init__()
        self.word_embeddings = TernaryEmbedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids, token_type_ids):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = (self.word_embeddings(input_ids) + self.position_embeddings(positions)
             + self.token_type_embeddings(token_type_ids))
        return self.LayerNorm(x)


class SelfAttention(nn.Module):
    """Q, K, V are GGM layers; Q K^T and A V stay in full precision."""

    def __init__(self, hidden_size, num_heads, N_factor):
        super().__init__()
        self.num_heads, self.head_size = num_heads, hidden_size // num_heads
        self.query = PackedLinearGGM(hidden_size, hidden_size, N_factor)
        self.key = PackedLinearGGM(hidden_size, hidden_size, N_factor)
        self.value = PackedLinearGGM(hidden_size, hidden_size, N_factor)
        self.attn_log_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.attn_base_scale = hidden_size * math.pi ** 2 / (4 * math.sqrt(self.head_size))

    def _heads(self, x):
        return x.view(*x.shape[:-1], self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x, mask):
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))
        scores = q @ k.transpose(-1, -2) * self.attn_log_scale.exp() * self.attn_base_scale
        probs = F.softmax(scores + mask, dim=-1)
        ctx = (probs @ v).permute(0, 2, 1, 3).contiguous()
        return ctx.view(*ctx.shape[:-2], -1)


class Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, N_factor):
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_heads, N_factor)
        self.attention_out = PackedLinearGGM(hidden_size, hidden_size, N_factor)
        self.attention_norm = BinaryNorm(hidden_size)
        self.intermediate = PackedLinearGGM(hidden_size, intermediate_size, N_factor)
        self.output = PackedLinearGGM(intermediate_size, hidden_size, N_factor)
        self.output_norm = BinaryNorm(hidden_size)

    def forward(self, x, mask):
        x = self.attention_norm(self.attention_out(self.attention(x, mask)) + x)
        return self.output_norm(self.output(F.relu(self.intermediate(x))) + x)


class BertGGM(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12, N_factor=1.0,
                 num_labels=2):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                                     layer_norm_eps)
        self.layers = nn.ModuleList(Layer(hidden_size, num_heads, intermediate_size, N_factor)
                                    for _ in range(num_layers))
        self.pooler = PackedLinearGGM(hidden_size, hidden_size, N_factor)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.classifier(F.relu(self.pooler(x[:, 0])))
