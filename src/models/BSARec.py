# -*- coding: UTF-8 -*-

import logging

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from utils.bsarec_layers import BSA_REC_INTERNAL_PADDING_IDX
from utils.bsarec_layers import BSARecEncoder
from utils.bsarec_layers import BSARecLayerNorm
from utils.bsarec_layers import bsarec_internal_logits_to_local_order
from utils.bsarec_layers import local_to_bsarec_internal_item_ids
from utils.bsarec_layers import mask_bsarec_internal_padding_logits
from utils.bsarec_layers import zero_bsarec_padding_embedding
from utils.constants import ITEM_ID, ITEM_SEQ, MAX_ITEM_SEQ_LEN


class BSARec(SequentialModel):
    reader = "BaseReader"
    runner = "BaseRunner"
    extra_log_args = [
        "emb_size",
        "num_layers",
        "num_heads",
        "dropout",
        "attention_dropout",
        "bsarec_alpha",
        "bsarec_c",
        "use_full_seq_loss",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size",
            type=int,
            default=64,
            help="Backbone hidden size used by BSARec.",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=2,
            help="Number of BSARec blocks.",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=2,
            help="Number of attention heads in the attention branch.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Hidden dropout probability used by BSARec.",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            default=0.5,
            help="Attention score dropout probability used by BSARec.",
        )
        parser.add_argument(
            "--hidden_act",
            type=str,
            default="gelu",
            help="Feed-forward activation.",
        )
        parser.add_argument(
            "--layer_norm_eps",
            type=float,
            default=1e-12,
            help="Layer normalization epsilon.",
        )
        parser.add_argument(
            "--initializer_range",
            type=float,
            default=0.02,
            help="Parameter initializer std.",
        )
        parser.add_argument(
            "--use_full_seq_loss",
            type=int,
            default=0,
            help="Whether to use full sequence loss (1) or last-position loss (0).",
        )
        parser.add_argument(
            "--bsarec_alpha",
            type=float,
            default=0.9,
            help="Mixing coefficient between frequency and attention branches.",
        )
        parser.add_argument(
            "--bsarec_c",
            type=int,
            default=3,
            help="Frequency truncation hyperparameter from upstream BSARec.",
        )
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)

        self.emb_size = args.emb_size
        self.hidden_size = self.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.use_full_seq_loss = int(args.use_full_seq_loss)
        self.bsarec_alpha = args.bsarec_alpha
        self.bsarec_c = args.bsarec_c

        self.local_padding_idx = self.item_num - 1
        self.internal_item_size = self.item_num
        self.internal_padding_idx = BSA_REC_INTERNAL_PADDING_IDX

        self._define_params()
        self.apply(self.init_weights)
        self._zero_padding_embeddings()

        logging.info("BSARec Configuration:")
        logging.info("  Hidden size: %s", self.hidden_size)
        logging.info("  Layers: %s", self.num_layers)
        logging.info("  Attention heads: %s", self.num_heads)
        logging.info("  Dropout: %s", self.dropout)
        logging.info("  Attention dropout: %s", self.attention_dropout)
        logging.info("  Alpha: %s", self.bsarec_alpha)
        logging.info("  c: %s", self.bsarec_c)
        logging.info("  Full-seq loss: %s", bool(self.use_full_seq_loss))

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BSARecLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _define_params(self):
        self.item_embeddings = nn.Embedding(
            self.internal_item_size,
            self.hidden_size,
            padding_idx=self.internal_padding_idx,
        )
        self.position_embeddings = nn.Embedding(MAX_ITEM_SEQ_LEN, self.hidden_size)
        self.LayerNorm = BSARecLayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.item_encoder = BSARecEncoder(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            attention_probs_dropout_prob=self.attention_dropout,
            hidden_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            alpha=self.bsarec_alpha,
            c=self.bsarec_c,
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def _zero_padding_embeddings(self) -> None:
        zero_bsarec_padding_embedding(self.item_embeddings)

    def _local_to_internal_item_ids(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        return local_to_bsarec_internal_item_ids(
            item_seq_ids,
            self.local_padding_idx,
        )

    def _local_targets_to_internal(self, target_items: torch.Tensor) -> torch.Tensor:
        return local_to_bsarec_internal_item_ids(
            target_items,
            self.local_padding_idx,
        )

    @staticmethod
    def _internal_logits_to_local_order(logits: torch.Tensor) -> torch.Tensor:
        return bsarec_internal_logits_to_local_order(logits)

    def _build_attention_mask(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (item_seq_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(
            torch.ones(attn_shape, device=item_seq_ids.device),
            diagonal=1,
        )
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.item_embeddings.weight.dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def add_position_embedding(self, internal_item_seq_ids: torch.Tensor) -> torch.Tensor:
        seq_length = internal_item_seq_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=internal_item_seq_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(internal_item_seq_ids)
        self._zero_padding_embeddings()
        item_embeddings = self.item_embeddings(internal_item_seq_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout_layer(sequence_emb)
        return sequence_emb

    def encode_sequence(
        self,
        item_seq_ids: torch.Tensor,
        all_sequence_output: bool = False,
    ):
        internal_item_seq_ids = self._local_to_internal_item_ids(item_seq_ids)
        extended_attention_mask = self._build_attention_mask(internal_item_seq_ids)
        sequence_emb = self.add_position_embedding(internal_item_seq_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=all_sequence_output,
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids = feed_dict[ITEM_SEQ]
        batch_size = item_seq_ids.size(0)
        seq_output = self.encode_sequence(item_seq_ids, all_sequence_output=False)

        out = {ITEM_ID: feed_dict[ITEM_ID], "sequence_output": seq_output}
        test_item_embs = self.item_embeddings.weight

        if stage == "train" and self.use_full_seq_loss:
            logits_internal = torch.matmul(seq_output, test_item_embs.transpose(0, 1))
            out["prediction_internal"] = logits_internal
            out["prediction"] = self._internal_logits_to_local_order(logits_internal)
            out["item_seq_ids"] = item_seq_ids
            out["padding_mask"] = item_seq_ids != self.local_padding_idx
            return out

        seq_len = item_seq_ids.size(1)
        gather_index = torch.full(
            (batch_size, 1, 1),
            seq_len - 1,
            dtype=torch.long,
            device=item_seq_ids.device,
        ).expand(-1, -1, self.hidden_size)
        seq_emb = torch.gather(seq_output, 1, gather_index).squeeze(1)
        logits_internal = torch.matmul(seq_emb, test_item_embs.transpose(0, 1))
        out["prediction_internal"] = logits_internal
        out["prediction"] = self._internal_logits_to_local_order(logits_internal)
        return out

    def loss(self, out_dict: dict) -> torch.Tensor:
        logits = out_dict["prediction_internal"]

        if self.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            target_item = out_dict[ITEM_ID]
            batch_size, seq_len, item_num = logits.size()

            labels = torch.cat(
                [
                    item_seq_ids[:, 1:],
                    target_item.unsqueeze(1),
                ],
                dim=1,
            )
            valid_mask = padding_mask.clone()
            if seq_len > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]

            labels_internal = self._local_targets_to_internal(labels)
            labels_internal[~valid_mask] = self.internal_padding_idx

            logits_flat = mask_bsarec_internal_padding_logits(logits).view(-1, item_num)
            labels_flat = labels_internal.view(-1)
            valid_mask_flat = valid_mask.view(-1)

            loss_flat = self.loss_fct(logits_flat, labels_flat)
            loss_flat = loss_flat * valid_mask_flat.float()
            return loss_flat.sum() / (valid_mask_flat.sum() + 1e-8)

        labels = out_dict[ITEM_ID]
        labels_internal = self._local_targets_to_internal(labels)
        logits_for_loss = mask_bsarec_internal_padding_logits(logits)
        return self.loss_fct(logits_for_loss, labels_internal).mean()
