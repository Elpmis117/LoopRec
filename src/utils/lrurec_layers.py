# -*- coding: UTF-8 -*-
"""Reusable LRURec layers and id-mapping helpers.

The LRURec model and LoopRec adapter use the same complex LRU block implementation.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LRUREC_INTERNAL_PADDING_IDX = 0


def local_to_lrurec_internal_item_ids(
    item_seq_ids: torch.Tensor,
    local_padding_idx: int,
) -> torch.Tensor:
    """Map local ids to official LRURec ids: padding=0, valid items=1..N."""

    return torch.where(
        item_seq_ids == local_padding_idx,
        item_seq_ids.new_full(item_seq_ids.shape, LRUREC_INTERNAL_PADDING_IDX),
        item_seq_ids + 1,
    )


def lrurec_internal_logits_to_local_order(logits: torch.Tensor) -> torch.Tensor:
    """Move internal padding class 0 to the final local padding column."""

    return torch.cat([logits[..., 1:], logits[..., :1]], dim=-1)


def truncated_normal_init_module(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 0.02,
    lower: float = -0.04,
    upper: float = 0.04,
) -> None:
    """Apply the official LRURec truncated-normal style initialization."""

    with torch.no_grad():
        l_bound = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
        u_bound = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0

        for name, param in module.named_parameters():
            if "layer_norm" in name or "params_log" in name:
                continue
            if torch.is_complex(param):
                param.real.uniform_(2 * l_bound - 1, 2 * u_bound - 1)
                param.imag.uniform_(2 * l_bound - 1, 2 * u_bound - 1)
                param.real.erfinv_()
                param.imag.erfinv_()
                param.real.mul_(std * math.sqrt(2.0)).add_(mean)
                param.imag.mul_(std * math.sqrt(2.0)).add_(mean)
            else:
                param.uniform_(2 * l_bound - 1, 2 * u_bound - 1)
                param.erfinv_()
                param.mul_(std * math.sqrt(2.0)).add_(mean)


def pad_sequence_to_power_of_two(
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Left-pad sequence tensors to the next power-of-two length for LRU scan."""

    seq_len = hidden_states.size(1)
    padded_len = 1 << int(math.ceil(math.log2(seq_len))) if seq_len > 1 else 1
    pad_len = padded_len - seq_len
    if pad_len <= 0:
        return hidden_states, valid_mask, seq_len

    hidden_states = F.pad(hidden_states, (0, 0, pad_len, 0, 0, 0))
    valid_mask = F.pad(valid_mask, (pad_len, 0, 0, 0))
    return hidden_states, valid_mask, seq_len


class LRURecBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        attention_dropout: float,
        r_min: float,
        r_max: float,
    ):
        super().__init__()
        self.lru_layer = LRURecLayer(
            d_model=hidden_size,
            dropout=attention_dropout,
            r_min=r_min,
            r_max=r_max,
        )
        self.feed_forward = LRURecFeedForward(
            d_model=hidden_size,
            d_ff=hidden_size * 4,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.lru_layer(hidden_states, valid_mask)
        return self.feed_forward(hidden_states)


class LRURecLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        use_bias: bool = True,
        r_min: float = 0.8,
        r_max: float = 0.99,
    ):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(
            torch.cfloat
        )
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(
            torch.cfloat
        )
        self.out_vector = nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    @staticmethod
    def lru_parallel(i, h, lamb, valid_mask, batch_size, seq_len, hidden_size):
        span = 2**i
        h = h.reshape(batch_size * seq_len // span, span, hidden_size)
        mask = valid_mask.reshape(batch_size * seq_len // span, span)
        h1, h2 = h[:, : span // 2], h[:, span // 2 :]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask[:, span // 2 - 1 : span // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma

        log2_len = int(np.ceil(np.log2(h.size(1)))) if h.size(1) > 1 else 0
        batch_size, seq_len, hidden_size = h.size(0), h.size(1), h.size(2)
        for i in range(log2_len):
            h, lamb = self.lru_parallel(
                i + 1,
                h,
                lamb,
                valid_mask,
                batch_size,
                seq_len,
                hidden_size,
            )
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)


class LRURecFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(hidden)) + x)
