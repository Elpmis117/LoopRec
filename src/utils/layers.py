# -*- coding: UTF-8 -*-
"""Transformer layers used by SASRec and the LoopRec SASRec adapter."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of heads (%d)"
                % (hidden_size, n_heads)
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape)

    def forward(self, input_tensor, attention_mask, kv_cache=None):
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)
        if kv_cache is not None:
            key = torch.cat([kv_cache["key"], key], dim=1)
            value = torch.cat([kv_cache["value"], value], dim=1)
        new_kv_cache = {"key": key, "value": value}

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.attn_dropout(self.softmax(attention_scores))

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, new_kv_cache


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

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
        return self.LayerNorm(hidden_states + input_tensor)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        attention_output, new_kv_cache = self.multi_head_attention(
            hidden_states,
            attention_mask,
            kv_cache,
        )
        return self.feed_forward(attention_output), new_kv_cache


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=False,
        kv_caches=None,
    ):
        all_encoder_layers = []
        present_kv_caches = []
        for i, layer_module in enumerate(self.layer):
            layer_kv_cache = None
            if kv_caches is not None and kv_caches[i] is not None:
                layer_kv_cache = kv_caches[i]
                layer_input = hidden_states[:, -1:, :]
            else:
                layer_input = hidden_states
            hidden_states, present_kv_cache = layer_module(
                layer_input,
                attention_mask,
                layer_kv_cache,
            )
            present_kv_caches.append(present_kv_cache)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, present_kv_caches


class SASRecWrapper(nn.Module):
    def __init__(self, transformer, hidden_size, dropout_prob=0.5, layer_norm_eps=1e-12):
        super().__init__()
        self.transformer = transformer
        self.n_layers = len(transformer.layer)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    @staticmethod
    def _prepare_padding_mask(item_seq_ids, padding_idx):
        return (item_seq_ids == padding_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _prepare_attention_mask(batch_size, seq_len, device, padding_mask):
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        return mask.masked_fill(padding_mask, -1e10)

    def forward(self, input_embs, item_seq_ids, padding_idx):
        batch_size, seq_len, _ = input_embs.size()
        padding_mask = self._prepare_padding_mask(item_seq_ids, padding_idx)
        attention_mask = self._prepare_attention_mask(
            batch_size,
            seq_len,
            input_embs.device,
            padding_mask,
        )
        input_embs = self.dropout(self.LayerNorm(input_embs))
        encoder_output, _ = self.transformer(
            input_embs,
            attention_mask,
            output_all_encoded_layers=False,
            kv_caches=[None] * self.n_layers,
        )
        return encoder_output[-1]


class PreLNMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of heads (%d)"
                % (hidden_size, n_heads)
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape)

    def forward(self, input_tensor, attention_mask, kv_cache=None):
        normed_input = self.LayerNorm(input_tensor)
        query = self.query_layer(normed_input)
        key = self.key_layer(normed_input)
        value = self.value_layer(normed_input)
        if kv_cache is not None and "key" in kv_cache and "value" in kv_cache:
            key = torch.cat([kv_cache["key"], key], dim=1)
            value = torch.cat([kv_cache["value"], value], dim=1)
        new_kv_cache = {"key": key, "value": value}

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.attn_dropout(self.softmax(attention_scores))

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        return hidden_states + input_tensor, new_kv_cache


class PreLNFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
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
        hidden_states = self.dense_1(self.LayerNorm(input_tensor))
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class PreLNTransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        self.multi_head_attention = PreLNMultiHeadAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = PreLNFeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        attention_output, new_kv_cache = self.multi_head_attention(
            hidden_states,
            attention_mask,
            kv_cache,
        )
        return self.feed_forward(attention_output), new_kv_cache
