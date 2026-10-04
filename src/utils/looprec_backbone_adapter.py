# -*- coding: UTF-8 -*-
"""SASRec, BSARec, NextItNet, and LRURec adapters for LoopRec."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from utils import layers
from utils.bsarec_layers import BSA_REC_INTERNAL_PADDING_IDX
from utils.bsarec_layers import BSARecBlock
from utils.bsarec_layers import BSARecLayerNorm
from utils.bsarec_layers import bsarec_internal_logits_to_local_order
from utils.bsarec_layers import local_to_bsarec_internal_item_ids
from utils.bsarec_layers import mask_bsarec_internal_padding_logits
from utils.bsarec_layers import zero_bsarec_padding_embedding
from utils.constants import ITEM_ID, ITEM_SEQ, MAX_ITEM_SEQ_LEN
from utils.lrurec_layers import LRUREC_INTERNAL_PADDING_IDX
from utils.lrurec_layers import LRURecBlock
from utils.lrurec_layers import local_to_lrurec_internal_item_ids
from utils.lrurec_layers import lrurec_internal_logits_to_local_order
from utils.lrurec_layers import pad_sequence_to_power_of_two
from utils.lrurec_layers import truncated_normal_init_module
from utils.looprec import LoopRecWrapper
from utils.nextitnet_layers import NextItNetResidualBlock
from utils.nextitnet_layers import build_dilation_schedule
from utils.nextitnet_layers import mask_nextitnet_padding_hidden
from utils.nextitnet_layers import zero_nextitnet_padding_embedding


@dataclass
class LoopRecBackboneInputs:
    item_seq_ids: torch.Tensor
    padding_mask: torch.Tensor
    input_embs: torch.Tensor
    batch_size: int


@dataclass
class LoopRecBackboneOutputs:
    prediction: torch.Tensor
    additional_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)


class LoopRecBackboneAdapter(ABC):
    def __init__(self, owner):
        self.owner = owner

    @staticmethod
    def _gather_last_sequence_hidden(
        hidden_states: torch.Tensor,
        item_seq_ids: torch.Tensor,
        hidden_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len = item_seq_ids.shape
        gather_index = torch.full(
            (batch_size, 1, 1),
            seq_len - 1,
            dtype=torch.long,
            device=item_seq_ids.device,
        ).expand(-1, -1, hidden_size)
        return torch.gather(hidden_states, 1, gather_index).squeeze(1)

    def _build_wrapper(
        self,
        shared_layers: nn.ModuleList,
        use_pre_norm: bool,
    ) -> LoopRecWrapper:
        owner = self.owner
        return LoopRecWrapper(
            transformer_layers=shared_layers,
            hidden_size=owner.hidden_size,
            num_recursion=owner.num_recursion,
            router_type=owner.looprec_router_type,
            router_temp=owner.looprec_router_temp,
            balancing=owner.looprec_balancing,
            use_z_loss=owner.looprec_use_z_loss,
            warmup_steps=owner.looprec_warmup_steps,
            kv_cache_strategy=owner.looprec_kv_cache_strategy,
            use_pre_norm=use_pre_norm,
            dropout_prob=owner.dropout_rate,
            layer_norm_eps=owner.layer_norm_eps,
            initializer_range=owner.initializer_range,
        )

    @abstractmethod
    def attach_backbone_modules(self) -> None:
        pass

    @abstractmethod
    def build_looprec_inputs(self, feed_dict: dict) -> LoopRecBackboneInputs:
        pass

    @abstractmethod
    def project_looprec_outputs(
        self,
        hidden_states: torch.Tensor,
        backbone_inputs: LoopRecBackboneInputs,
        feed_dict: dict,
        stage: str,
    ) -> LoopRecBackboneOutputs:
        pass

    @abstractmethod
    def compute_task_loss(self, out_dict: dict) -> torch.Tensor:
        pass

    def after_owner_init_weights(self) -> None:
        pass


class SASRecLoopRecBackboneAdapter(LoopRecBackboneAdapter):
    def attach_backbone_modules(self) -> None:
        owner = self.owner
        owner.item_emb = nn.Embedding(
            owner.item_num,
            owner.emb_size,
            padding_idx=owner.item_num - 1,
        )
        owner.pos_emb = nn.Embedding(
            MAX_ITEM_SEQ_LEN + 1,
            owner.emb_size,
            padding_idx=MAX_ITEM_SEQ_LEN,
        )
        trm_layer = layers.PreLNTransformerLayer(
            n_heads=owner.num_heads,
            hidden_size=owner.hidden_size,
            intermediate_size=owner.inner_size,
            hidden_dropout_prob=owner.dropout_rate,
            attn_dropout_prob=owner.dropout_rate,
            hidden_act=owner.hidden_act,
            layer_norm_eps=owner.layer_norm_eps,
        )
        shared_layers = nn.ModuleList(
            [copy.deepcopy(trm_layer) for _ in range(owner.num_layers)]
        )
        owner.looprec_wrapper = self._build_wrapper(shared_layers, use_pre_norm=True)
        owner.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def build_looprec_inputs(self, feed_dict: dict) -> LoopRecBackboneInputs:
        owner = self.owner
        item_seq_ids = feed_dict[ITEM_SEQ]
        padding_mask = item_seq_ids != (owner.item_num - 1)
        valid_pos_ids = torch.cumsum(padding_mask.long(), dim=1) - 1
        pos_ids = torch.where(padding_mask, valid_pos_ids, MAX_ITEM_SEQ_LEN)
        input_embs = owner.item_emb(item_seq_ids) + owner.pos_emb(pos_ids)
        return LoopRecBackboneInputs(
            item_seq_ids=item_seq_ids,
            padding_mask=padding_mask,
            input_embs=input_embs,
            batch_size=item_seq_ids.size(0),
        )

    def project_looprec_outputs(
        self,
        hidden_states: torch.Tensor,
        backbone_inputs: LoopRecBackboneInputs,
        feed_dict: dict,
        stage: str,
    ) -> LoopRecBackboneOutputs:
        del feed_dict
        owner = self.owner
        item_embs = owner.item_emb.weight
        if stage == "train" and owner.use_full_seq_loss:
            logits = torch.matmul(hidden_states, item_embs.transpose(0, 1))
            return LoopRecBackboneOutputs(
                prediction=logits,
                additional_outputs={
                    "item_seq_ids": backbone_inputs.item_seq_ids,
                    "padding_mask": backbone_inputs.padding_mask,
                },
            )

        seq_emb = self._gather_last_sequence_hidden(
            hidden_states,
            backbone_inputs.item_seq_ids,
            owner.hidden_size,
        )
        logits = torch.matmul(seq_emb, item_embs.transpose(0, 1))
        return LoopRecBackboneOutputs(prediction=logits)

    def compute_task_loss(self, out_dict: dict) -> torch.Tensor:
        owner = self.owner
        logits = out_dict["prediction"]
        if owner.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            labels = torch.cat([item_seq_ids[:, 1:], out_dict[ITEM_ID].unsqueeze(1)], dim=1)
            valid_mask = padding_mask.clone()
            if valid_mask.size(1) > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]
            loss_flat = owner.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_flat = loss_flat * valid_mask.view(-1).float()
            return loss_flat.sum() / (valid_mask.sum() + 1e-8)
        return owner.loss_fct(logits, out_dict[ITEM_ID]).mean()


class _LoopRecBSARecBlockAdapter(nn.Module):
    def __init__(self, block: BSARecBlock):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        del kv_cache
        return self.block(hidden_states, attention_mask), None


class BSARecLoopRecBackboneAdapter(LoopRecBackboneAdapter):
    def attach_backbone_modules(self) -> None:
        owner = self.owner
        owner.local_padding_idx = owner.item_num - 1
        owner.internal_item_size = owner.item_num
        owner.internal_padding_idx = BSA_REC_INTERNAL_PADDING_IDX
        owner.item_embeddings = nn.Embedding(
            owner.internal_item_size,
            owner.hidden_size,
            padding_idx=owner.internal_padding_idx,
        )
        owner.position_embeddings = nn.Embedding(MAX_ITEM_SEQ_LEN, owner.hidden_size)
        bsarec_block = _LoopRecBSARecBlockAdapter(
            BSARecBlock(
                hidden_size=owner.hidden_size,
                num_attention_heads=owner.num_heads,
                attention_probs_dropout_prob=owner.attention_dropout,
                hidden_dropout_prob=owner.dropout_rate,
                hidden_act=owner.hidden_act,
                layer_norm_eps=owner.layer_norm_eps,
                alpha=owner.bsarec_alpha,
                c=owner.bsarec_c,
            )
        )
        shared_layers = nn.ModuleList(
            [copy.deepcopy(bsarec_block) for _ in range(owner.num_layers)]
        )
        owner.looprec_wrapper = self._build_wrapper(shared_layers, use_pre_norm=False)
        owner.looprec_wrapper.input_layer_norm = BSARecLayerNorm(
            owner.hidden_size,
            eps=owner.layer_norm_eps,
        )
        owner.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def _local_to_internal_item_ids(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        return local_to_bsarec_internal_item_ids(
            item_seq_ids,
            self.owner.local_padding_idx,
        )

    def _local_targets_to_internal(self, target_items: torch.Tensor) -> torch.Tensor:
        return local_to_bsarec_internal_item_ids(
            target_items,
            self.owner.local_padding_idx,
        )

    def after_owner_init_weights(self) -> None:
        zero_bsarec_padding_embedding(self.owner.item_embeddings)

    def build_looprec_inputs(self, feed_dict: dict) -> LoopRecBackboneInputs:
        owner = self.owner
        item_seq_ids = feed_dict[ITEM_SEQ]
        internal_item_seq_ids = self._local_to_internal_item_ids(item_seq_ids)
        seq_len = item_seq_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_ids)
        zero_bsarec_padding_embedding(owner.item_embeddings)
        input_embs = owner.item_embeddings(internal_item_seq_ids)
        input_embs = input_embs + owner.position_embeddings(position_ids)
        return LoopRecBackboneInputs(
            item_seq_ids=item_seq_ids,
            padding_mask=item_seq_ids != owner.local_padding_idx,
            input_embs=input_embs,
            batch_size=item_seq_ids.size(0),
        )

    def project_looprec_outputs(
        self,
        hidden_states: torch.Tensor,
        backbone_inputs: LoopRecBackboneInputs,
        feed_dict: dict,
        stage: str,
    ) -> LoopRecBackboneOutputs:
        del feed_dict
        owner = self.owner
        item_embs = owner.item_embeddings.weight
        if stage == "train" and owner.use_full_seq_loss:
            logits_internal = torch.matmul(hidden_states, item_embs.transpose(0, 1))
            return LoopRecBackboneOutputs(
                prediction=bsarec_internal_logits_to_local_order(logits_internal),
                additional_outputs={
                    "prediction_internal": logits_internal,
                    "item_seq_ids": backbone_inputs.item_seq_ids,
                    "padding_mask": backbone_inputs.padding_mask,
                },
            )

        seq_emb = self._gather_last_sequence_hidden(
            hidden_states,
            backbone_inputs.item_seq_ids,
            owner.hidden_size,
        )
        logits_internal = torch.matmul(seq_emb, item_embs.transpose(0, 1))
        return LoopRecBackboneOutputs(
            prediction=bsarec_internal_logits_to_local_order(logits_internal),
            additional_outputs={"prediction_internal": logits_internal},
        )

    def compute_task_loss(self, out_dict: dict) -> torch.Tensor:
        owner = self.owner
        logits_internal = out_dict["prediction_internal"]
        if owner.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            labels = torch.cat([item_seq_ids[:, 1:], out_dict[ITEM_ID].unsqueeze(1)], dim=1)
            valid_mask = padding_mask.clone()
            if valid_mask.size(1) > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]
            labels_internal = self._local_targets_to_internal(labels)
            labels_internal[~valid_mask] = owner.internal_padding_idx
            logits_flat = mask_bsarec_internal_padding_logits(logits_internal).view(
                -1,
                logits_internal.size(-1),
            )
            loss_flat = owner.loss_fct(logits_flat, labels_internal.view(-1))
            loss_flat = loss_flat * valid_mask.view(-1).float()
            return loss_flat.sum() / (valid_mask.sum() + 1e-8)

        labels_internal = self._local_targets_to_internal(out_dict[ITEM_ID])
        logits_for_loss = mask_bsarec_internal_padding_logits(logits_internal)
        return owner.loss_fct(logits_for_loss, labels_internal).mean()


class _LoopRecNextItNetBlockAdapter(nn.Module):
    def __init__(self, block: NextItNetResidualBlock):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        del kv_cache
        if hidden_states.size(1) != MAX_ITEM_SEQ_LEN:
            raise ValueError(
                "NextItNet requires full-length sequences inside LoopRec; "
                "use the tail-window full-sequence protocol."
            )
        valid_mask = None
        if attention_mask is not None and attention_mask.dim() == 4:
            valid_mask = ~(attention_mask[:, 0].diagonal(dim1=-2, dim2=-1) < 0)
        return self.block(hidden_states, valid_mask=valid_mask), None


class NextItNetLoopRecBackboneAdapter(LoopRecBackboneAdapter):
    def attach_backbone_modules(self) -> None:
        owner = self.owner
        owner.local_padding_idx = owner.item_num - 1
        owner.internal_item_size = owner.item_num
        owner.item_embeddings = nn.Embedding(
            owner.internal_item_size,
            owner.hidden_size,
            padding_idx=owner.local_padding_idx,
        )
        owner.input_dropout = nn.Dropout(owner.dropout_rate)
        dilation_schedule = build_dilation_schedule(
            owner.nextitnet_dilations,
            owner.num_layers,
            owner.nextitnet_block_num,
        )
        owner.nextitnet_dilation_schedule = dilation_schedule
        shared_layers = nn.ModuleList(
            [
                _LoopRecNextItNetBlockAdapter(
                    NextItNetResidualBlock(
                        hidden_size=owner.hidden_size,
                        kernel_size=owner.nextitnet_kernel_size,
                        dilation=dilation,
                    )
                )
                for dilation in dilation_schedule
            ]
        )
        owner.looprec_wrapper = self._build_wrapper(shared_layers, use_pre_norm=False)
        owner.loss_fct = nn.CrossEntropyLoss(reduction="none")
        owner.final_layer = nn.Linear(owner.hidden_size, owner.hidden_size)

    def after_owner_init_weights(self) -> None:
        zero_nextitnet_padding_embedding(self.owner.item_embeddings)

    def build_looprec_inputs(self, feed_dict: dict) -> LoopRecBackboneInputs:
        owner = self.owner
        zero_nextitnet_padding_embedding(owner.item_embeddings)
        item_seq_ids = feed_dict[ITEM_SEQ]
        input_embs = owner.input_dropout(owner.item_embeddings(item_seq_ids))
        return LoopRecBackboneInputs(
            item_seq_ids=item_seq_ids,
            padding_mask=item_seq_ids != owner.local_padding_idx,
            input_embs=input_embs,
            batch_size=item_seq_ids.size(0),
        )

    def project_looprec_outputs(
        self,
        hidden_states: torch.Tensor,
        backbone_inputs: LoopRecBackboneInputs,
        feed_dict: dict,
        stage: str,
    ) -> LoopRecBackboneOutputs:
        del feed_dict
        owner = self.owner
        hidden_states = owner.final_layer(hidden_states)
        hidden_states = mask_nextitnet_padding_hidden(
            hidden_states,
            backbone_inputs.padding_mask,
        )
        item_embs = owner.item_embeddings.weight
        if stage == "train" and owner.use_full_seq_loss:
            logits_internal = torch.matmul(hidden_states, item_embs.transpose(0, 1))
            return LoopRecBackboneOutputs(
                prediction=logits_internal,
                additional_outputs={
                    "prediction_internal": logits_internal,
                    "item_seq_ids": backbone_inputs.item_seq_ids,
                    "padding_mask": backbone_inputs.padding_mask,
                    "sequence_output": hidden_states,
                },
            )

        seq_emb = self._gather_last_sequence_hidden(
            hidden_states,
            backbone_inputs.item_seq_ids,
            owner.hidden_size,
        )
        logits_internal = torch.matmul(seq_emb, item_embs.transpose(0, 1))
        return LoopRecBackboneOutputs(
            prediction=logits_internal,
            additional_outputs={
                "prediction_internal": logits_internal,
                "sequence_output": hidden_states,
            },
        )

    def _residual_block_reg_loss(self) -> torch.Tensor:
        owner = self.owner
        loss = owner.item_embeddings.weight.new_tensor(0.0)
        if owner.nextitnet_reg_weight <= 0.0:
            return loss
        for name, param in owner.looprec_wrapper.recursive_layers.named_parameters():
            if name.endswith("weight"):
                loss = loss + torch.norm(param, p=2)
        return owner.nextitnet_reg_weight * loss

    def compute_task_loss(self, out_dict: dict) -> torch.Tensor:
        owner = self.owner
        logits_internal = out_dict["prediction_internal"]
        if owner.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            labels = torch.cat([item_seq_ids[:, 1:], out_dict[ITEM_ID].unsqueeze(1)], dim=1)
            valid_mask = padding_mask.clone()
            if valid_mask.size(1) > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]
            labels = labels.clone()
            labels[~valid_mask] = 0
            loss_flat = owner.loss_fct(
                logits_internal.view(-1, logits_internal.size(-1)),
                labels.view(-1),
            )
            loss_flat = loss_flat * valid_mask.view(-1).float()
            loss = loss_flat.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = owner.loss_fct(logits_internal, out_dict[ITEM_ID]).mean()
        return loss + self._residual_block_reg_loss()


class _LoopRecLRURecBlockAdapter(nn.Module):
    def __init__(self, block: LRURecBlock):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        del kv_cache
        if attention_mask is not None and attention_mask.dim() == 4:
            valid_mask = ~(attention_mask[:, 0].diagonal(dim1=-2, dim2=-1) < 0)
        else:
            valid_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        hidden_states, valid_mask, seq_len = pad_sequence_to_power_of_two(
            hidden_states,
            valid_mask,
        )
        return self.block(hidden_states, valid_mask)[:, -seq_len:], None


class LRURecLoopRecBackboneAdapter(LoopRecBackboneAdapter):
    def attach_backbone_modules(self) -> None:
        owner = self.owner
        owner.local_padding_idx = owner.item_num - 1
        owner.internal_padding_idx = LRUREC_INTERNAL_PADDING_IDX
        owner.internal_item_size = owner.item_num
        owner.item_embeddings = nn.Embedding(owner.internal_item_size, owner.hidden_size)
        owner.input_layer_norm = nn.LayerNorm(owner.hidden_size)
        owner.embed_dropout = nn.Dropout(owner.dropout_rate)
        owner.output_bias = nn.Parameter(torch.zeros(owner.internal_item_size))
        lru_block = _LoopRecLRURecBlockAdapter(
            LRURecBlock(
                hidden_size=owner.hidden_size,
                dropout=owner.dropout_rate,
                attention_dropout=owner.attention_dropout,
                r_min=owner.lrurec_r_min,
                r_max=owner.lrurec_r_max,
            )
        )
        shared_layers = nn.ModuleList(
            [copy.deepcopy(lru_block) for _ in range(owner.num_layers)]
        )
        owner.looprec_wrapper = self._build_wrapper(shared_layers, use_pre_norm=False)
        owner.looprec_wrapper.input_layer_norm = nn.Identity()
        owner.looprec_wrapper.input_dropout = nn.Identity()
        owner.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def after_owner_init_weights(self) -> None:
        owner = self.owner
        truncated_normal_init_module(owner.item_embeddings, std=owner.initializer_range)
        truncated_normal_init_module(
            owner.looprec_wrapper.recursive_layers,
            std=owner.initializer_range,
        )
        nn.init.trunc_normal_(
            owner.output_bias,
            mean=0.0,
            std=owner.initializer_range,
            a=-0.04,
            b=0.04,
        )

    def _local_to_internal_item_ids(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        return local_to_lrurec_internal_item_ids(
            item_seq_ids,
            self.owner.local_padding_idx,
        )

    def _local_targets_to_internal(self, target_items: torch.Tensor) -> torch.Tensor:
        return local_to_lrurec_internal_item_ids(
            target_items,
            self.owner.local_padding_idx,
        )

    def build_looprec_inputs(self, feed_dict: dict) -> LoopRecBackboneInputs:
        owner = self.owner
        item_seq_ids = feed_dict[ITEM_SEQ]
        internal_item_seq_ids = self._local_to_internal_item_ids(item_seq_ids)
        input_embs = owner.item_embeddings(internal_item_seq_ids)
        input_embs = owner.input_layer_norm(owner.embed_dropout(input_embs))
        return LoopRecBackboneInputs(
            item_seq_ids=item_seq_ids,
            padding_mask=item_seq_ids != owner.local_padding_idx,
            input_embs=input_embs,
            batch_size=item_seq_ids.size(0),
        )

    def project_looprec_outputs(
        self,
        hidden_states: torch.Tensor,
        backbone_inputs: LoopRecBackboneInputs,
        feed_dict: dict,
        stage: str,
    ) -> LoopRecBackboneOutputs:
        del feed_dict
        owner = self.owner
        item_embs = owner.item_embeddings.weight
        if stage == "train" and owner.use_full_seq_loss:
            logits_internal = torch.matmul(hidden_states, item_embs.transpose(0, 1))
            logits_internal = logits_internal + owner.output_bias
            return LoopRecBackboneOutputs(
                prediction=lrurec_internal_logits_to_local_order(logits_internal),
                additional_outputs={
                    "prediction_internal": logits_internal,
                    "item_seq_ids": backbone_inputs.item_seq_ids,
                    "padding_mask": backbone_inputs.padding_mask,
                    "sequence_output": hidden_states,
                },
            )

        seq_emb = self._gather_last_sequence_hidden(
            hidden_states,
            backbone_inputs.item_seq_ids,
            owner.hidden_size,
        )
        logits_internal = torch.matmul(seq_emb, item_embs.transpose(0, 1))
        logits_internal = logits_internal + owner.output_bias
        return LoopRecBackboneOutputs(
            prediction=lrurec_internal_logits_to_local_order(logits_internal),
            additional_outputs={
                "prediction_internal": logits_internal,
                "sequence_output": hidden_states,
            },
        )

    def compute_task_loss(self, out_dict: dict) -> torch.Tensor:
        owner = self.owner
        logits_internal = out_dict["prediction_internal"]
        if owner.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            labels = torch.cat([item_seq_ids[:, 1:], out_dict[ITEM_ID].unsqueeze(1)], dim=1)
            valid_mask = torch.cat(
                [
                    padding_mask[:, 1:],
                    out_dict[ITEM_ID].ne(owner.local_padding_idx).unsqueeze(1),
                ],
                dim=1,
            )
            labels_internal = self._local_targets_to_internal(labels)
            labels_internal[~valid_mask] = owner.internal_padding_idx
            loss_flat = owner.loss_fct(
                logits_internal.view(-1, logits_internal.size(-1)),
                labels_internal.view(-1),
            )
            loss_flat = loss_flat * valid_mask.view(-1).float()
            return loss_flat.sum() / (valid_mask.sum() + 1e-8)

        labels_internal = self._local_targets_to_internal(out_dict[ITEM_ID])
        return owner.loss_fct(logits_internal, labels_internal).mean()


def build_looprec_backbone_adapter(owner, looprec_backbone: str) -> LoopRecBackboneAdapter:
    if looprec_backbone == "sasrec":
        return SASRecLoopRecBackboneAdapter(owner)
    if looprec_backbone == "bsarec":
        return BSARecLoopRecBackboneAdapter(owner)
    if looprec_backbone == "nextitnet":
        return NextItNetLoopRecBackboneAdapter(owner)
    if looprec_backbone == "lrurec":
        return LRURecLoopRecBackboneAdapter(owner)
    raise ValueError(f"Unsupported looprec_backbone={looprec_backbone!r}.")
