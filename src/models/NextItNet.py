# -*- coding: UTF-8 -*-

import logging

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from utils.constants import ITEM_ID, ITEM_SEQ
from utils.nextitnet_layers import NextItNetEncoder
from utils.nextitnet_layers import mask_nextitnet_padding_hidden
from utils.nextitnet_layers import zero_nextitnet_padding_embedding


class NextItNet(SequentialModel):
    reader = "BaseReader"
    runner = "BaseRunner"
    extra_log_args = [
        "emb_size",
        "num_layers",
        "dropout",
        "nextitnet_dilations",
        "nextitnet_kernel_size",
        "nextitnet_block_num",
        "nextitnet_reg_weight",
        "nextitnet_output_head",
        "use_full_seq_loss",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size",
            type=int,
            default=64,
            help="Backbone hidden size used by NextItNet.",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=2,
            help="Number of NextItNet residual blocks.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Embedding dropout probability used by NextItNet.",
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
            default=1,
            help="Whether to use full sequence loss (1) or last-position loss (0).",
        )
        parser.add_argument(
            "--nextitnet_dilations",
            type=str,
            default="1,2,4,8",
            help="Comma-separated causal dilation pattern for NextItNet blocks.",
        )
        parser.add_argument(
            "--nextitnet_kernel_size",
            type=int,
            default=3,
            help="Causal convolution kernel size for NextItNet blocks.",
        )
        parser.add_argument(
            "--nextitnet_block_num",
            type=int,
            default=1,
            help="Number of times to repeat the dilation pattern before truncation.",
        )
        parser.add_argument(
            "--nextitnet_reg_weight",
            type=float,
            default=0.0,
            help="Extra L2 penalty weight on NextItNet residual block weights.",
        )
        parser.add_argument(
            "--nextitnet_output_head",
            type=str,
            default="tied_embedding",
            choices=("tied_embedding", "untied_linear"),
            help=(
                "Output classifier for NextItNet. tied_embedding keeps the "
                "existing RecBole-style embedding dot-product head; "
                "untied_linear uses an independent full-softmax projection."
            ),
        )
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)

        self.emb_size = args.emb_size
        self.hidden_size = self.emb_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.initializer_range = args.initializer_range
        self.use_full_seq_loss = int(args.use_full_seq_loss)
        self.nextitnet_dilations = args.nextitnet_dilations
        self.nextitnet_kernel_size = int(args.nextitnet_kernel_size)
        self.nextitnet_block_num = int(args.nextitnet_block_num)
        self.nextitnet_reg_weight = float(args.nextitnet_reg_weight)
        self.nextitnet_output_head = getattr(
            args,
            "nextitnet_output_head",
            "tied_embedding",
        )

        self.local_padding_idx = self.item_num - 1
        self.internal_item_size = self.item_num

        self._define_params()
        self.apply(self.init_weights)
        self._zero_padding_embeddings()

        logging.info("NextItNet Configuration:")
        logging.info("  Hidden size: %s", self.hidden_size)
        logging.info("  Layers: %s", self.num_layers)
        logging.info("  Dropout: %s", self.dropout)
        logging.info("  Dilations: %s", self.nextitnet_dilations)
        logging.info("  Dilation schedule: %s", self.item_encoder.dilation_schedule)
        logging.info("  Kernel size: %s", self.nextitnet_kernel_size)
        logging.info("  Block num: %s", self.nextitnet_block_num)
        logging.info("  Reg weight: %s", self.nextitnet_reg_weight)
        logging.info("  Output head: %s", self.nextitnet_output_head)
        logging.info("  Full-seq loss: %s", bool(self.use_full_seq_loss))

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _define_params(self):
        self.item_embeddings = nn.Embedding(
            self.internal_item_size,
            self.hidden_size,
            padding_idx=self.local_padding_idx,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.item_encoder = NextItNetEncoder(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dilations=self.nextitnet_dilations,
            kernel_size=self.nextitnet_kernel_size,
            block_num=self.nextitnet_block_num,
        )
        self.final_layer = nn.Linear(self.hidden_size, self.hidden_size)
        if self.nextitnet_output_head == "untied_linear":
            self.output_layer = nn.Linear(self.hidden_size, self.item_num)
        else:
            self.output_layer = None
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def _zero_padding_embeddings(self) -> None:
        zero_nextitnet_padding_embedding(self.item_embeddings)

    def _local_to_internal_item_ids(self, item_seq_ids: torch.Tensor) -> torch.Tensor:
        return item_seq_ids

    def add_item_embedding(self, internal_item_seq_ids: torch.Tensor) -> torch.Tensor:
        self._zero_padding_embeddings()
        return self.dropout_layer(self.item_embeddings(internal_item_seq_ids))

    def encode_sequence(
        self,
        item_seq_ids: torch.Tensor,
        all_sequence_output: bool = False,
    ):
        internal_item_seq_ids = self._local_to_internal_item_ids(item_seq_ids)
        valid_mask = item_seq_ids != self.local_padding_idx
        sequence_emb = self.add_item_embedding(internal_item_seq_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            output_all_encoded_layers=all_sequence_output,
            valid_mask=valid_mask,
        )
        if all_sequence_output:
            return [
                mask_nextitnet_padding_hidden(self.final_layer(layer), valid_mask)
                for layer in item_encoded_layers
            ]
        return mask_nextitnet_padding_hidden(
            self.final_layer(item_encoded_layers[-1]),
            valid_mask,
        )

    def _residual_block_reg_loss(self) -> torch.Tensor:
        loss = self.item_embeddings.weight.new_tensor(0.0)
        if self.nextitnet_reg_weight <= 0.0:
            return loss
        for name, param in self.item_encoder.named_parameters():
            if name.endswith("weight"):
                loss = loss + torch.norm(param, p=2)
        return self.nextitnet_reg_weight * loss

    def compute_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        if self.nextitnet_output_head == "untied_linear":
            return self.output_layer(torch.relu(sequence_output))
        test_item_embs = self.item_embeddings.weight
        return torch.matmul(sequence_output, test_item_embs.transpose(0, 1))

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        del epoch
        item_seq_ids = feed_dict[ITEM_SEQ]
        batch_size = item_seq_ids.size(0)
        sequence_output = self.encode_sequence(item_seq_ids, all_sequence_output=False)

        out = {ITEM_ID: feed_dict[ITEM_ID], "sequence_output": sequence_output}

        if stage == "train" and self.use_full_seq_loss:
            logits_internal = self.compute_logits(sequence_output)
            out["prediction_internal"] = logits_internal
            out["prediction"] = logits_internal
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
        logits_internal = self.compute_logits(seq_emb)
        out["prediction_internal"] = logits_internal
        out["prediction"] = logits_internal
        return out

    def loss(self, out_dict: dict) -> torch.Tensor:
        logits_internal = out_dict["prediction_internal"]

        if self.use_full_seq_loss and "item_seq_ids" in out_dict:
            item_seq_ids = out_dict["item_seq_ids"]
            padding_mask = out_dict["padding_mask"]
            target_item = out_dict[ITEM_ID]
            _, seq_len, item_num = logits_internal.size()

            labels = torch.cat(
                [item_seq_ids[:, 1:], target_item.unsqueeze(1)],
                dim=1,
            )
            valid_mask = padding_mask.clone()
            if seq_len > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]

            labels_internal = labels.clone()
            labels_internal[~valid_mask] = 0

            logits_flat = logits_internal.view(-1, item_num)
            labels_flat = labels_internal.view(-1)
            valid_mask_flat = valid_mask.view(-1)

            loss_flat = self.loss_fct(logits_flat, labels_flat)
            loss_flat = loss_flat * valid_mask_flat.float()
            loss = loss_flat.sum() / (valid_mask_flat.sum() + 1e-8)
        else:
            labels = out_dict[ITEM_ID]
            loss = self.loss_fct(logits_internal, labels).mean()

        return loss + self._residual_block_reg_loss()
