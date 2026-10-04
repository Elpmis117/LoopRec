# -*- coding: UTF-8 -*-

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BSA_REC_INTERNAL_PADDING_IDX = 0
BSA_REC_PADDING_LOGIT_MASK_VALUE = -10000.0


def local_to_bsarec_internal_item_ids(
    item_seq_ids: torch.Tensor,
    local_padding_idx: int,
) -> torch.Tensor:
    """Map repo-local item ids to BSARec ids with internal padding at 0."""

    return torch.where(
        item_seq_ids == local_padding_idx,
        item_seq_ids.new_full(item_seq_ids.shape, BSA_REC_INTERNAL_PADDING_IDX),
        item_seq_ids + 1,
    )


def bsarec_internal_logits_to_local_order(logits: torch.Tensor) -> torch.Tensor:
    """Move BSARec internal padding class from index 0 to the local final column."""

    return torch.cat([logits[..., 1:], logits[..., :1]], dim=-1)


def mask_bsarec_internal_padding_logits(logits: torch.Tensor) -> torch.Tensor:
    """Mask the internal padding class before CE without producing infinities."""

    masked_logits = logits.clone()
    masked_logits[..., BSA_REC_INTERNAL_PADDING_IDX] = BSA_REC_PADDING_LOGIT_MASK_VALUE
    return masked_logits


def zero_bsarec_padding_embedding(embedding: nn.Embedding) -> None:
    """Keep BSARec internal padding row from leaking into the FFT branch."""

    padding_idx = embedding.padding_idx
    if padding_idx is None:
        return
    with torch.no_grad():
        embedding.weight[padding_idx].zero_()


class BSARecLayerNorm(nn.Module):
    """TensorFlow-style LayerNorm used by the upstream BSARec repo."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BSARecFeedForward(nn.Module):
    """Feed-forward block matching the upstream BSARec implementation."""

    def __init__(self, hidden_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        inner_size = 4 * hidden_size
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = BSARecLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    @staticmethod
    def gelu(x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def get_hidden_act(self, act):
        act2fn = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return act2fn[act]

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BSARecMultiHeadAttention(nn.Module):
    """Multi-head attention block matching upstream BSARec."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BSARecFrequencyLayer(nn.Module):
    """Frequency filter branch from upstream BSARec."""

    def __init__(self, hidden_size, hidden_dropout_prob, c, layer_norm_eps):
        super().__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = BSARecLayerNorm(hidden_size, eps=layer_norm_eps)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        _, seq_len, _ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")

        low_pass = x.clone()
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm="ortho")
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BSARecLayer(nn.Module):
    """One BSARec layer: frequency branch + attention branch."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
        alpha,
        c,
    ):
        super().__init__()
        self.filter_layer = BSARecFrequencyLayer(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            c=c,
            layer_norm_eps=layer_norm_eps,
        )
        self.attention_layer = BSARecMultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + (1.0 - self.alpha) * gsp
        return hidden_states


class BSARecBlock(nn.Module):
    """One BSARec block: mixed layer + feed-forward."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
        alpha,
        c,
    ):
        super().__init__()
        self.layer = BSARecLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            alpha=alpha,
            c=c,
        )
        self.feed_forward = BSARecFeedForward(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecEncoder(nn.Module):
    """Stack of BSARec blocks."""

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
        alpha,
        c,
    ):
        super().__init__()
        block = BSARecBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            alpha=alpha,
            c=c,
        )
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
