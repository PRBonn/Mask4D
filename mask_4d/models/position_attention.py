import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionAttention(nn.Module):
    """Position-aware Mask Attention."""

    def __init__(self, num_hiddens, num_heads, dropout, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.in_proj_weight = nn.parameter.Parameter(
            torch.empty(3 * num_hiddens, num_hiddens)
        )
        self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * num_hiddens))
        self.out_proj = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.hd = num_hiddens

    def forward(self, query, key, value, attn_mask, key_padding_mask, kernel):
        if key_padding_mask:
            attn_mask = attn_mask + key_padding_mask
        queries = self.transpose_qkv(
            F.linear(
                query, self.in_proj_weight[: self.hd, :], self.in_proj_bias[: self.hd]
            )
        )
        keys = self.transpose_qkv(
            F.linear(
                key,
                self.in_proj_weight[self.hd : -self.hd, :],
                self.in_proj_bias[self.hd : -self.hd],
            )
        )
        values = self.transpose_qkv(
            F.linear(
                value,
                self.in_proj_weight[-self.hd :, :],
                self.in_proj_bias[-self.hd :],
            )
        )
        output = self.attention(queries, keys, values, attn_mask, kernel)
        output_concat = self.transpose_output(output)
        return self.out_proj(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None, kernel=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = scores + kernel.permute(0, 2, 1)
        self.attention_weights = masked_softmax(scores, mask)
        return torch.bmm(self.dropout(self.attention_weights), values)


def masked_softmax(X, mask):
    """Perform softmax operation by masking elements on the last axis."""
    if mask is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        mask = torch.zeros_like(mask, dtype=X.dtype).masked_fill_(mask, float("-inf"))
        X = X + mask
        return nn.functional.softmax(X, dim=-1)
