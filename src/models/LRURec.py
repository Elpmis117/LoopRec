# -*- coding: UTF-8 -*-

import logging

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from utils.constants import ITEM_ID, ITEM_SEQ
from utils.lrurec_layers import LRUREC_INTERNAL_PADDING_IDX
from utils.lrurec_layers import LRURecBlock
from utils.lrurec_layers import local_to_lrurec_internal_item_ids
from utils.lrurec_layers import lrurec_internal_logits_to_local_order
from utils.lrurec_layers import pad_sequence_to_power_of_two
from utils.lrurec_layers import truncated_normal_init_module


class LRURec(SequentialModel):
    reader = "BaseReader"
    runner = "BaseRunner"
    extra_log_args = [
        "emb_size",
        "num_layers",
        "dropout",
        "attention_dropout",
        "use_full_seq_loss",
        "lrurec_r_min",
        "lrurec_r_max",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size",
            type=int,
            default=64,
            help="Backbone hidden size used by LRURec.",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=2,
            help="Number of LRURec blocks.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Embedding and feed-forward dropout probability used by LRURec.",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            default=0.2,
            help="LRU recurrent output dropout probability.",
        )
        parser.add_argument(
            "--initializer_range",
            type=float,
            default=0.02,
            help="Truncated normal initializer std.",
        )
        parser.add_argument(
            "--use_full_seq_loss",
            type=int,
            default=1,
            help="Whether to use full sequence loss (1) or last-position loss (0).",
        )
        parser.add_argument(
            "--lrurec_r_min",
            type=float,
            default=0.8,
            help="Minimum recurrent radius used by the official LRU layer init.",
        )
        parser.add_argument(
            "--lrurec_r_max",
            type=float,
            default=0.99,
            help="Maximum recurrent radius used by the official LRU layer init.",
        )
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)

        self.emb_size = args.emb_size
        self.hidden_size = self.emb_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.initializer_range = args.initializer_range
        self.use_full_seq_loss = int(args.use_full_seq_loss)
        self.lrurec_r_min = float(args.lrurec_r_min)
        self.lrurec_r_max = float(args.lrurec_r_max)

        self.local_padding_idx = self.item_num - 1
        self.internal_padding_idx = LRUREC_INTERNAL_PADDING_IDX
        self.internal_item_size = self.item_num

        self._define_params()
        self.truncated_normal_init(std=self.initializer_range)

        logging.info("LRURec Configuration:")
        logging.info("  Hidden size: %s", self.hidden_size)
        logging.info("  Layers: %s", self.num_layers)
        logging.info("  Dropout: %s", self.dropout)
        logging.info("  Attention dropout: %s", self.attention_dropout)
        logging.info("  Full-seq loss: %s", bool(self.use_full_seq_loss))
        logging.info("  r_min/r_max: %s/%s", self.lrurec_r_min, self.lrurec_r_max)

    def _define_params(self):
        self.item_embeddings = nn.Embedding(
            self.internal_item_size,
            self.hidden_size,
        )
        self.input_layer_norm = nn.LayerNorm(self.hidden_size)
        self.embed_dropout = nn.Dropout(self.dropout)
        self.lru_blocks = nn.ModuleList(
            [
                LRURecBlock(
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    r_min=self.lrurec_r_min,
                    r_max=self.lrurec_r_max,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_bias = nn.Parameter(torch.zeros(self.internal_item_size))
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def truncated_normal_init(
        self,
        mean: float = 0.0,
        std: float = 0.02,
        lower: float = -0.04,
        upper: float = 0.04,
    ) -> None:
        truncated_normal_init_module(
            self,
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
        )

    def _local_to_internal_item_ids(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        return local_to_lrurec_internal_item_ids(
            item_seq_ids,
            self.local_padding_idx,
        )

    def _local_targets_to_internal(self, target_items: torch.Tensor) -> torch.Tensor:
        return local_to_lrurec_internal_item_ids(
            target_items,
            self.local_padding_idx,
        )

    @staticmethod
    def _internal_logits_to_local_order(logits: torch.Tensor) -> torch.Tensor:
        return lrurec_internal_logits_to_local_order(logits)

    def add_item_embedding(self, internal_item_seq_ids: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.item_embeddings(internal_item_seq_ids)
        return self.input_layer_norm(self.embed_dropout(item_embeddings))

    def encode_internal_sequence(self, internal_item_seq_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = internal_item_seq_ids > self.internal_padding_idx
        hidden_states = self.add_item_embedding(internal_item_seq_ids)

        hidden_states, valid_mask, seq_len = pad_sequence_to_power_of_two(
            hidden_states,
            valid_mask,
        )

        for block in self.lru_blocks:
            hidden_states = block(hidden_states, valid_mask)

        return hidden_states[:, -seq_len:]

    def encode_sequence(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        internal_item_seq_ids = self._local_to_internal_item_ids(item_seq_ids)
        return self.encode_internal_sequence(internal_item_seq_ids)

    def compute_internal_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return (
            torch.matmul(sequence_output, self.item_embeddings.weight.transpose(0, 1))
            + self.output_bias
        )

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        del epoch
        item_seq_ids = feed_dict[ITEM_SEQ]
        batch_size = item_seq_ids.size(0)
        sequence_output = self.encode_sequence(item_seq_ids)
        out = {ITEM_ID: feed_dict[ITEM_ID], "sequence_output": sequence_output}

        if stage == "train" and self.use_full_seq_loss:
            logits_internal = self.compute_internal_logits(sequence_output)
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
        seq_emb = torch.gather(sequence_output, 1, gather_index).squeeze(1)
        logits_internal = self.compute_internal_logits(seq_emb)
        out["prediction_internal"] = logits_internal
        out["prediction"] = self._internal_logits_to_local_order(logits_internal)
        return out

    def loss(self, out_dict: dict) -> torch.Tensor:
        logits_internal = out_dict["prediction_internal"]

        if self.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            target_item = out_dict[ITEM_ID]
            _, seq_len, item_num = logits_internal.size()

            labels = torch.cat(
                [
                    item_seq_ids[:, 1:],
                    target_item.unsqueeze(1),
                ],
                dim=1,
            )
            valid_mask = torch.cat(
                [
                    padding_mask[:, 1:],
                    target_item.ne(self.local_padding_idx).unsqueeze(1),
                ],
                dim=1,
            )

            labels_internal = self._local_targets_to_internal(labels)
            labels_internal[~valid_mask] = self.internal_padding_idx

            logits_flat = logits_internal.view(-1, item_num)
            labels_flat = labels_internal.view(-1)
            valid_mask_flat = valid_mask.view(-1)

            loss_flat = self.loss_fct(logits_flat, labels_flat)
            loss_flat = loss_flat * valid_mask_flat.float()
            return loss_flat.sum() / (valid_mask_flat.sum() + 1e-8)

        labels_internal = self._local_targets_to_internal(out_dict[ITEM_ID])
        return self.loss_fct(logits_internal, labels_internal).mean()
