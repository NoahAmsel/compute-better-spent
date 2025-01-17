"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from sympy import factorint
from einops import rearrange

from .cola_nn import cola_init

class BilinearBTT_W_K_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (c, d, Hbs)
        self.weight = nn.Parameter(torch.randn(shape))
        self.weight.d_in = shape[1]
        self.weight.fan_in_dims = 1
        self.weight.d_out = shape[2]
        cola_init(self.weight, zero_init=False, init_method='µP')

    def forward(self, x):
        '''x.shape = (c, BT, d)'''

        if True:
            RMS_W_K = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-2)
            d_out = self.weight.size(-1)
            init_scale_W_K = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_K_normed = self.weight / max(1, RMS_W_K / init_scale_W_K)

        # output.shape = (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        return torch.bmm(x, W_K_normed)

class BilinearBTT_W_Q_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (Hb, a, cs)
        self.weight = nn.Parameter(torch.randn(shape))
        self.weight.d_in = shape[2]
        self.weight.fan_in_dims = 2
        self.weight.d_out = shape[1]
        cola_init(self.weight, zero_init=False, init_method='µP')

    def forward(self, x):
        '''x.shape = (Hb, cs, BT)'''

        if True:
            RMS_W_Q = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-1)
            d_out = self.weight.size(-2)
            init_scale_W_Q = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_Q_normed = self.weight / max(1, RMS_W_Q / init_scale_W_Q)

        # output.shape = (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        return torch.bmm(W_Q_normed, x)


class BilinearBTTAttention(nn.Module):

    def __init__(self, dim, heads, dim_head, btt_tt_dim, btt_tt_rank, bilinear_btt_muP_attn_logits_scaling, dropout=0., use_bias=True):
        super().__init__()
        assert dim % heads == 0
        assert dim == dim_head * heads
        # assert config.split_qkv==True
        # assert config.manual_disable_flash_att==True
        # assert config.do_qk_ln==True

        self.split_qkv = True  # config.split_qkv 
        self.do_qk_ln = True  # config.do_qk_ln
        self.manual_disable_flash_att = True # config.manual_disable_flash_att
        self.link_function = "softmax"

        self.bilinear_btt_muP_attn_logits_scaling = bilinear_btt_muP_attn_logits_scaling

        self.n_head = heads
        self.d_model = dim
        self.d_qk_head = dim_head
        self.dropout = dropout

        self.btt_tt_dim = btt_tt_dim
        self.btt_tt_rank = btt_tt_rank

        # define BTT shape
        self.a, self.b = self.c, self.d = self.factorize(self.d_model, self.btt_tt_dim)
        self.s = self.btt_tt_rank

        # define BTT projection matrices
        self.bilinear_btt_wk = BilinearBTT_W_K_Projection((self.c, self.d, self.n_head*self.b*self.s))
        self.bilinear_btt_wq = BilinearBTT_W_Q_Projection((self.n_head*self.b, self.a, self.c*self.s))

        if self.do_qk_ln:
            self.bilinear_btt_ln_PLPRXT = nn.LayerNorm(dim)

        # W_V, W_O Projections
        self.c_attn_v = nn.Linear(dim, dim, bias=use_bias)
        self.c_attn_v.weight.d_in = dim
        self.c_attn_v.weight.fan_in_dims = 0
        self.c_attn_v.weight.d_out = dim
        cola_init(self.c_attn_v.weight, zero_init=False, init_method='µP')

        self.c_proj = nn.Linear(dim, dim, bias=use_bias)
        self.c_proj.weight.d_in = dim
        self.c_proj.weight.fan_in_dims = 0
        self.c_proj.weight.d_out = dim
        cola_init(self.c_proj.weight, zero_init=True, init_method='µP')

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))

    def factorize(self, x, n=2):
        # Get prime factors and their counts
        prime_factors = factorint(x)

        # Initialize the n integers
        numbers = [1] * n

        # Distribute the prime factors
        for prime, count in prime_factors.items():
            for _ in range(count):
                # Find the number with the smallest product to assign the prime factor
                min_index = min(range(n), key=lambda i: numbers[i])
                numbers[min_index] *= prime

        # return in ascending order
        return sorted(numbers)

    def bilinear_btt_einsum(self, x):        
        '''
        (Pdb) !torch.sum(att_einsum!=att)
        '''
        B, T, C = x.size()

        # att_einsum.shape = (c, BT, d)
        att_einsum = rearrange(x, "B T (c d) -> c (B T) d", B=B, T=T, c=self.c, d=self.d)

        # (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        att_einsum = self.bilinear_btt_wk(att_einsum)

        # att_einsum.shape = (Hb, cs, BT)
        att_einsum = rearrange(att_einsum, "c (B T) (H b s) -> (H b) (c s) (B T)", B=B, T=T, H=self.n_head, s=self.s, b=self.b, c=self.c)

        # (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        att_einsum = self.bilinear_btt_wq(att_einsum)

        # att_einsum.shape (B, HT, ab)
        att_einsum = rearrange(att_einsum, "(H b) a (B T) -> B (H T) (a b)", B=B, T=T, H=self.n_head, a=self.a, b=self.b)

        # att_einsum.shape = (B, T, HT) | Final Contraction
        att_einsum = torch.bmm(x, self.bilinear_btt_ln_PLPRXT(att_einsum).transpose(-2, -1))

        if self.bilinear_btt_muP_attn_logits_scaling:
            att_einsum = att_einsum * (1.0 / self.d_model) # muP attn logits scaling
        else:
            att_einsum = att_einsum * (1.0 / math.sqrt(self.d_model)) # SP attn logits scaling

        # att_einsum.shape = (B, H, T, T)
        att_einsum = rearrange(att_einsum, "B T (H N) -> B H T N", B=B, T=T, H=self.n_head, N=T)

        return att_einsum

    def forward(self, x, verify=False):
        B, T, C = x.size()

        # compute v_projection
        v = self.c_attn_v(x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        ############################################ Bilinear BTT ############################################

        # att.shape = (c, BT, d)
        att = x.reshape(B*T, self.c, self.d).permute(1, 0, 2)

        # (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        att = self.bilinear_btt_wk(att)

        # att.shape = (Hb, cs, BT)
        att = att.reshape(self.c, B*T, self.n_head*self.b, self.s).permute(0, 3, 1, 2).reshape(self.c*self.s, B*T, self.n_head*self.b).permute(2, 0, 1)

        # (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        att = self.bilinear_btt_wq(att)

        # att.shape (B, HT, ab)
        att = att.reshape(self.n_head, self.b, self.a, B, T).permute(3, 0, 4, 2, 1).reshape(B, self.n_head*T, self.a*self.b)

        # att.shape = (B, T, HT) = (B, T, ab) * (B, ab, HT)
        att = torch.bmm(x, self.bilinear_btt_ln_PLPRXT(att).transpose(-2, -1))

        if self.bilinear_btt_muP_attn_logits_scaling:
            att = att * (1.0 / self.d_model) # muP attn logits scaling
        else:
            att = att * (1.0 / math.sqrt(self.d_model)) # SP attn logits scaling
        
        # att.shape = (B, H, T, T)
        att = att.reshape(B, T, self.n_head, T).permute(0, 2, 1, 3)

        # if verify:
        #     att_einsum = self.bilinear_btt_einsum(x)
        #     assert torch.sum(att!=att_einsum)==0
        #     print("pass assertion")
        

        ############################################ Bilinear BTT ############################################

        if self.link_function == "softmax":
            # no bias here...
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
        elif self.link_function == "identity":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
        else:
            raise ValueError 
        
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
