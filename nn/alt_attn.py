import torch
import torch.nn as nn
from .cola_nn import cola_init


class LowRankQK(nn.Module):
    def __init__(self, heads, dimension, rank, attn_mult=1):
        super().__init__()
        self.attn_mult = attn_mult
        self.rank = rank

        self.L = nn.Parameter(torch.empty(heads, rank, dimension))
        self.L.d_in = dimension
        self.L.fan_in_dims = 2
        self.L.d_out = rank
        # Follow the `Attention` class, which zero_init's WQ and WO
        cola_init(self.L, zero_init=True, init_method='µP')

        self.R = nn.Parameter(torch.empty(heads, rank, dimension))
        self.R.d_in = dimension
        self.R.fan_in_dims = 2
        self.R.d_out = rank
        cola_init(self.R, zero_init=False, init_method='µP')

    def forward(self, query_x, key_x):
        scale = self.attn_mult * 8 / self.rank  # μP prescribes this scaling, 8 for backward compatibility at dim_head=64
        # NOTE when seq_length^2 > dim*rank, we should move `*scale` inside
        return torch.einsum("bLd,hrd,hrD,bMD->bhLM", query_x, self.L, self.R, key_x) * scale


class LowRankVO(nn.Module):
    def __init__(self, heads, dimension, rank, use_bias=True):
        super().__init__()
        self.VO_bias = nn.Parameter(torch.zeros(dimension)) if use_bias else None
        self.rank = rank

        self.L = nn.Parameter(torch.empty(heads, rank, dimension))
        self.L.d_in = dimension
        self.L.fan_in_dims = 2
        self.L.d_out = rank
        cola_init(self.L, zero_init=False, init_method='µP')

        self.R = nn.Parameter(torch.empty(heads, rank, dimension))
        self.R.d_in = rank
        self.R.fan_in_dims = 1
        self.R.d_out = dimension
        # Follow the `Attention` class, which zero_init's WQ and WO
        cola_init(self.R, zero_init=True, init_method='µP')

    def forward(self, value_x, attn):
        out = torch.einsum("bhLM,bMD,hrD,hrd->bLd", attn, value_x, self.L, self.R)
        if self.VO_bias is not None:
            out += self.VO_bias
        return out


# class MultiheadLRPD(MultiheadLowRank):
#     def __init__(self, heads, dimension, rank):
#         super().__init__(heads=heads, dimension=dimension, rank=rank)
#         self.D = nn.Parameter(torch.empty(heads, dimension))
#         self.D.d_in = 1  # double check this
#         self.D.d_out = 1
#         cola_init(self.D, zero_init=False, init_method='µP')


class StructuredAttention(nn.Module):
    def __init__(self, dim, QK, VO, dropout=0., fixup=False, causal=False):
        super().__init__()
        self.QK = QK
        self.VO = VO
        self.causal = causal

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # a scaler multiplier
        self.out_scalar = nn.Parameter(torch.ones(1)) if fixup else None

    def forward(self, x):
        x = self.norm(x)

        dots = self.QK(x, x)  # b h n n

        # attention mask
        if self.causal:
            mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1)
            dots = dots.masked_fill(mask == 1, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = self.VO(x, attn)
        if self.out_scalar is not None:
            out = out * self.out_scalar
        return out

