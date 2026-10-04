# -*- coding: UTF-8 -*-
"""NextItNet layers for the local native port.

The residual block follows the RecBole ``ResidualBlock_b`` structure: two
left-padded causal dilated convolutions with residual connection.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_dilations(value: str | Iterable[int]) -> list[int]:
    """Parse a comma-separated dilation pattern."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("nextitnet_dilations must contain at least one integer.")
        dilations = [int(part) for part in parts]
    else:
        dilations = [int(part) for part in value]
    if not dilations or any(dilation < 1 for dilation in dilations):
        raise ValueError("nextitnet_dilations must be positive integers.")
    return dilations


def build_dilation_schedule(
    dilation_pattern: str | Iterable[int],
    num_layers: int,
    block_num: int = 1,
) -> list[int]:
    """Build the residual-block dilation schedule.

    ``num_layers`` is the total number of residual blocks exposed through the
    existing config/launcher vocabulary. ``block_num`` repeats the base pattern
    before truncation and is kept for compatibility with RecBole naming.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be >= 1 for NextItNet.")
    if block_num < 1:
        raise ValueError("nextitnet_block_num must be >= 1.")

    pattern = parse_dilations(dilation_pattern)
    schedule = pattern * block_num
    repeats = math.ceil(num_layers / len(schedule))
    return (schedule * repeats)[:num_layers]


def zero_nextitnet_padding_embedding(embedding: nn.Embedding) -> None:
    """Keep NextItNet padding embeddings from leaking into causal convs."""

    padding_idx = embedding.padding_idx
    if padding_idx is None:
        return
    with torch.no_grad():
        embedding.weight[padding_idx].zero_()


def mask_nextitnet_padding_hidden(
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Zero padding hidden states so causal convs cannot read them."""

    if valid_mask is None:
        return hidden_states
    return hidden_states * valid_mask.to(hidden_states.dtype).unsqueeze(-1)


class NextItNetResidualBlock(nn.Module):
    """Residual block (b) from the NextItNet PyTorch reference lineage."""

    def __init__(self, hidden_size: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("nextitnet_kernel_size must be >= 1.")
        if dilation < 1:
            raise ValueError("NextItNet dilation must be >= 1.")

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size),
            padding=0,
            dilation=dilation,
        )
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size),
            padding=0,
            dilation=dilation * 2,
        )
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-8)

    def _causal_conv_pad(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        x = x.permute(0, 2, 1).unsqueeze(2)
        left_pad = (self.kernel_size - 1) * dilation
        return F.pad(x, (left_pad, 0, 0, 0))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = mask_nextitnet_padding_hidden(x, valid_mask)
        out = self._causal_conv_pad(x, self.dilation)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)
        out = F.relu(self.ln1(out))
        out = mask_nextitnet_padding_hidden(out, valid_mask)

        out = self._causal_conv_pad(out, self.dilation * 2)
        out = self.conv2(out).squeeze(2).permute(0, 2, 1)
        out = F.relu(self.ln2(out))
        out = out + x
        return mask_nextitnet_padding_hidden(out, valid_mask)


class NextItNetEncoder(nn.Module):
    """Stack of causal dilated convolution residual blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dilations: str | Iterable[int] = "1,2,4,8",
        kernel_size: int = 3,
        block_num: int = 1,
    ):
        super().__init__()
        dilation_schedule = build_dilation_schedule(dilations, num_layers, block_num)
        self.dilation_schedule = dilation_schedule
        self.blocks = nn.ModuleList(
            [
                NextItNetResidualBlock(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in dilation_schedule
            ]
        )

    def forward(
        self,
        hidden_states,
        output_all_encoded_layers=False,
        valid_mask: torch.Tensor | None = None,
    ):
        hidden_states = mask_nextitnet_padding_hidden(hidden_states, valid_mask)
        all_encoder_layers = [hidden_states]
        for block in self.blocks:
            hidden_states = block(hidden_states, valid_mask=valid_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
