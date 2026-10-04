# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from models.BaseModel import SequentialModel
from utils import layers
from utils.constants import *


class SASRec(SequentialModel):
    reader = "BaseReader"
    runner = "BaseRunner"
    extra_log_args = ["emb_size", "num_layers", "num_heads", "dropout"]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size", type=int, default=256, help="Size of embeddings"
        )
        parser.add_argument(
            "--num_layers", type=int, default=2, help="Number of self-attention layers"
        )
        parser.add_argument(
            "--num_heads", type=int, default=2, help="Number of attention heads"
        )
        parser.add_argument(
            "--inner_size", type=int, default=300, help="Size of inner hidden layers"
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout probability for each deep layer",
        )
        parser.add_argument(
            "--hidden_act",
            type=str,
            default="gelu",
            help="Activation function of hidden layers",
        )
        parser.add_argument(
            "--layer_norm_eps",
            type=float,
            default=1e-12,
            help="Layer normalization epsilon",
        )
        parser.add_argument(
            "--initializer_range",
            type=float,
            default=0.02,
            help="Initializer range for parameters",
        )
        parser.add_argument(
            "--use_full_seq_loss",
            type=int,
            default=1,
            help="Whether to use full sequence loss (1) or only last position (0)",
        )
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.hidden_size = self.emb_size  # same as emb_size
        self.inner_size = args.inner_size
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.dropout = args.dropout
        self.use_full_seq_loss = args.use_full_seq_loss
        self._define_params()
        self.apply(self.init_weights)
        
        # Log configuration
        logging.info(f"SASRec Configuration:")
        logging.info(f"  - Layers: {self.num_layers}")
        logging.info(f"  - Hidden size: {self.hidden_size}")
        logging.info(f"  - Attention heads: {self.num_heads}")
        logging.info(f"  - FFN size: {self.inner_size}")
        logging.info(f"  - Full sequence loss: {bool(self.use_full_seq_loss)}")

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _define_params(self):
        self.item_emb = nn.Embedding(
            self.item_num, self.emb_size, padding_idx=self.item_num - 1
        )
        self.pos_emb = nn.Embedding(
            MAX_ITEM_SEQ_LEN + 1, self.emb_size, padding_idx=MAX_ITEM_SEQ_LEN
        )
        self.trm_encoder = layers.TransformerEncoder(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.model = layers.SASRecWrapper(
            self.trm_encoder,
            self.hidden_size,
            dropout_prob=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids, item_seq_len = feed_dict[ITEM_SEQ], feed_dict[ITEM_SEQ_LEN]
        batch_size = item_seq_ids.size(0)

        # Left Padding for item_seq_ids
        # Create position ids based on actual positions (excluding padding)
        padding_mask_bool = item_seq_ids != (self.item_num - 1)
        valid_pos_ids = torch.cumsum(padding_mask_bool.long(), dim=1) - 1
        pos_ids = torch.where(padding_mask_bool, valid_pos_ids, MAX_ITEM_SEQ_LEN)

        # Get embeddings
        item_embs = self.item_emb(item_seq_ids)
        pos_embs = self.pos_emb(pos_ids)
        input_embs = item_embs + pos_embs

        # Forward through SASRec model wrapper
        seq_output = self.model(input_embs, item_seq_ids, self.item_num - 1)
        # seq_output: [batch_size, seq_len, hidden_size]

        if stage == "train" and self.use_full_seq_loss:
            # Full sequence loss: predict next item at each position
            # Compute logits for all positions
            test_item_embs = self.item_emb.weight  # [item_num, hidden_size]
            logits = torch.matmul(seq_output, test_item_embs.transpose(0, 1))
            # logits: [batch_size, seq_len, item_num]
            
            return {
                "prediction": logits,
                "item_seq_ids": item_seq_ids,
                "padding_mask": padding_mask_bool,
                ITEM_ID: feed_dict[ITEM_ID],
            }
        else:
            # Evaluation mode or single-position loss: only use last position
            # For LEFT-PADDED sequences, the last valid item is always at position seq_len-1
            seq_len = item_seq_ids.size(1)
            gather_index = torch.full(
                (batch_size, 1, 1), seq_len - 1,
                dtype=torch.long, device=item_seq_ids.device
            ).expand(-1, -1, self.hidden_size)
            seq_emb = torch.gather(seq_output, 1, gather_index).squeeze(1)  # [batch_size, hidden_size]

            # Compute prediction scores
            test_item_embs = self.item_emb.weight
            logits = torch.matmul(seq_emb, test_item_embs.transpose(0, 1))

            return {
                "prediction": logits,
                ITEM_ID: feed_dict[ITEM_ID],
            }

    def loss(self, out_dict: dict) -> torch.Tensor:
        logits = out_dict["prediction"]
        
        if self.use_full_seq_loss and "item_seq_ids" in out_dict:
            # Full sequence loss
            # logits: [batch_size, seq_len, item_num]
            # We need to predict position i+1 from position i
            
            item_seq_ids = out_dict["item_seq_ids"]  # [batch_size, seq_len]
            padding_mask = out_dict["padding_mask"]  # [batch_size, seq_len]
            target_item = out_dict[ITEM_ID]  # [batch_size] - the actual target item
            
            batch_size, seq_len, item_num = logits.size()
            
            # Shift labels: position i predicts position i+1
            # The last position should predict the TARGET item (from dataset)
            labels = torch.cat([
                item_seq_ids[:, 1:],  # items from position 1 to end
                target_item.unsqueeze(1)  # use target item for last position
            ], dim=1)  # [batch_size, seq_len]
            
            # Mask for valid predictions
            # Position i can predict if:
            # 1. Position i is not padding (padding_mask[:, i] == True)
            # 2. For positions 0 to seq_len-2: position i+1 is not padding
            # 3. For position seq_len-1: valid only if position is not padding
            valid_mask = padding_mask.clone()
            
            # For left-padded sequences, ensure next position is valid
            if seq_len > 1:
                valid_mask[:, :-1] = valid_mask[:, :-1] & padding_mask[:, 1:]
            # Last position: keeps its validity from padding_mask
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, item_num)  # [batch_size * seq_len, item_num]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]
            valid_mask_flat = valid_mask.view(-1)  # [batch_size * seq_len]
            
            # Compute loss only on valid positions
            loss_flat = self.loss_fct(logits_flat, labels_flat)  # [batch_size * seq_len]
            loss_flat = loss_flat * valid_mask_flat.float()
            
            # Average over valid positions
            loss = loss_flat.sum() / (valid_mask_flat.sum() + 1e-8)
            
        else:
            # Single position loss for last-item training/evaluation.
            labels = out_dict[ITEM_ID]
            loss = self.loss_fct(logits, labels).mean()
        
        return loss
