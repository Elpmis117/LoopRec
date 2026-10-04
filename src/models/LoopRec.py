# -*- coding: UTF-8 -*-
"""LoopRec: item-level adaptive recursion for sequential recommendation.

Uses ordered-exit routing over shared SASRec, BSARec, NextItNet, or LRURec blocks.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from utils.constants import ITEM_ID
from utils.looprec import LoopRecOutput
from utils.looprec_backbone_adapter import build_looprec_backbone_adapter


class LoopRec(SequentialModel):
    reader = "BaseReader"
    runner = "BaseRunner"
    extra_log_args = [
        "emb_size",
        "num_layers",
        "num_recursion",
        "num_heads",
        "dropout",
        "looprec_backbone",
        "looprec_router_type",
        "looprec_router_temp",
        "looprec_router_input_source",
        "looprec_balancing",
        "looprec_balancing_coeff",
        "looprec_use_z_loss",
        "looprec_z_loss_coeff",
        "looprec_warmup_epochs",
        "looprec_kv_cache_strategy",
        "looprec_gating",
        "looprec_localdepth_parametrization",
        "looprec_localdepth_training_mode",
        "attention_dropout",
        "bsarec_alpha",
        "bsarec_c",
        "nextitnet_dilations",
        "nextitnet_kernel_size",
        "nextitnet_block_num",
        "nextitnet_reg_weight",
        "lrurec_r_min",
        "lrurec_r_max",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", type=int, default=256)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--inner_size", type=int, default=300)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--attention_dropout", type=float, default=0.5)
        parser.add_argument("--hidden_act", type=str, default="gelu")
        parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--use_full_seq_loss", type=int, default=1)

        parser.add_argument(
            "--looprec_backbone",
            dest="looprec_backbone",
            type=str,
            default="sasrec",
            choices=["sasrec", "bsarec", "nextitnet", "lrurec"],
        )
        parser.add_argument("--num_recursion", type=int, default=3)
        parser.add_argument(
            "--looprec_router_formulation",
            dest="looprec_router_formulation",
            type=str,
            default="local_depth_classifier",
            choices=["local_depth_classifier"],
        )
        parser.add_argument(
            "--looprec_router_type",
            dest="looprec_router_type",
            type=str,
            default="mlp",
            choices=["linear", "mlp"],
        )
        parser.add_argument(
            "--looprec_router_func",
            dest="looprec_router_func",
            type=str,
            default="softmax",
            choices=["softmax"],
        )
        parser.add_argument(
            "--looprec_router_temp",
            dest="looprec_router_temp",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--looprec_router_input_source",
            dest="looprec_router_input_source",
            type=str,
            default="first_block_output",
            choices=["first_block_output"],
        )
        parser.add_argument(
            "--looprec_localdepth_parametrization",
            dest="looprec_localdepth_parametrization",
            type=str,
            default="ordered_exit",
            choices=["ordered_exit"],
        )
        parser.add_argument(
            "--looprec_localdepth_training_mode",
            dest="looprec_localdepth_training_mode",
            type=str,
            default="selected_only_st",
            choices=["selected_only_st"],
        )
        parser.add_argument(
            "--looprec_balancing",
            dest="looprec_balancing",
            type=str,
            default="loss",
            choices=["loss", "none"],
        )
        parser.add_argument(
            "--looprec_balancing_coeff",
            dest="looprec_balancing_coeff",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--looprec_use_z_loss",
            dest="looprec_use_z_loss",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--looprec_z_loss_coeff",
            dest="looprec_z_loss_coeff",
            type=float,
            default=1e-3,
        )
        parser.add_argument(
            "--looprec_warmup_steps",
            dest="looprec_warmup_steps",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--looprec_warmup_epochs",
            dest="looprec_warmup_epochs",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--looprec_kv_cache_strategy",
            dest="looprec_kv_cache_strategy",
            type=str,
            default="recursion_wise",
            choices=["recursion_wise", "kv_sharing"],
        )
        parser.add_argument(
            "--looprec_gating",
            dest="looprec_gating",
            type=str,
            default="weighted",
            choices=["weighted"],
        )
        parser.add_argument(
            "--looprec_exited_token_memory",
            dest="looprec_exited_token_memory",
            type=str,
            default="active_only",
            choices=["active_only"],
        )

        parser.add_argument("--bsarec_alpha", type=float, default=0.9)
        parser.add_argument("--bsarec_c", type=int, default=3)
        parser.add_argument("--nextitnet_dilations", type=str, default="1,2,4,8")
        parser.add_argument("--nextitnet_kernel_size", type=int, default=3)
        parser.add_argument("--nextitnet_block_num", type=int, default=1)
        parser.add_argument("--nextitnet_reg_weight", type=float, default=0.0)
        parser.add_argument("--lrurec_r_min", type=float, default=0.8)
        parser.add_argument("--lrurec_r_max", type=float, default=0.99)

        return SequentialModel.parse_model_args(parser)

    @staticmethod
    def _validate_mainline_scope(args):
        if args.num_recursion < 1:
            raise ValueError("num_recursion must be positive.")
        if args.looprec_router_formulation != "local_depth_classifier":
            raise ValueError("looprec_router_formulation must be local_depth_classifier.")
        if args.looprec_router_input_source != "first_block_output":
            raise ValueError("looprec_router_input_source must be first_block_output.")
        if args.looprec_localdepth_parametrization != "ordered_exit":
            raise ValueError("looprec_localdepth_parametrization must be ordered_exit.")
        if args.looprec_localdepth_training_mode != "selected_only_st":
            raise ValueError("looprec_localdepth_training_mode must be selected_only_st.")
        if args.looprec_router_func != "softmax":
            raise ValueError("looprec_router_func must be softmax.")
        if args.looprec_gating != "weighted":
            raise ValueError("looprec_gating must be weighted.")
        if args.looprec_exited_token_memory != "active_only":
            raise ValueError("looprec_exited_token_memory must be active_only.")
        if args.looprec_backbone in {"nextitnet", "lrurec"}:
            if args.looprec_kv_cache_strategy != "kv_sharing":
                raise ValueError(
                    f"looprec_backbone={args.looprec_backbone} requires "
                    "looprec_kv_cache_strategy=kv_sharing."
                )
        elif args.looprec_kv_cache_strategy != "recursion_wise":
            raise ValueError(
                f"looprec_backbone={args.looprec_backbone} requires "
                "looprec_kv_cache_strategy=recursion_wise."
            )

    def __init__(self, args, corpus):
        self._validate_mainline_scope(args)

        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.hidden_size = self.emb_size
        self.inner_size = args.inner_size
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.dropout_rate = args.dropout
        self.attention_dropout = args.attention_dropout
        self.use_full_seq_loss = args.use_full_seq_loss

        self.looprec_backbone = args.looprec_backbone
        self.num_recursion = args.num_recursion
        self.looprec_router_formulation = args.looprec_router_formulation
        self.looprec_router_type = args.looprec_router_type
        self.looprec_router_func = args.looprec_router_func
        self.looprec_router_temp = args.looprec_router_temp
        self.looprec_router_input_source = args.looprec_router_input_source
        self.looprec_balancing = args.looprec_balancing
        self.looprec_balancing_coeff = args.looprec_balancing_coeff
        self.looprec_use_z_loss = bool(args.looprec_use_z_loss)
        self.looprec_z_loss_coeff = args.looprec_z_loss_coeff
        self.looprec_warmup_steps = args.looprec_warmup_steps
        self.looprec_warmup_epochs = args.looprec_warmup_epochs
        self.looprec_kv_cache_strategy = args.looprec_kv_cache_strategy
        self.looprec_gating = args.looprec_gating
        self.looprec_localdepth_parametrization = args.looprec_localdepth_parametrization
        self.looprec_localdepth_training_mode = args.looprec_localdepth_training_mode
        self.looprec_exited_token_memory = args.looprec_exited_token_memory
        self._looprec_return_routing_info_in_infer = False

        self.bsarec_alpha = args.bsarec_alpha
        self.bsarec_c = args.bsarec_c
        self.nextitnet_dilations = args.nextitnet_dilations
        self.nextitnet_kernel_size = int(args.nextitnet_kernel_size)
        self.nextitnet_block_num = int(args.nextitnet_block_num)
        self.nextitnet_reg_weight = float(args.nextitnet_reg_weight)
        self.lrurec_r_min = float(args.lrurec_r_min)
        self.lrurec_r_max = float(args.lrurec_r_max)

        self._looprec_backbone_adapter = None
        self._define_params()
        self.apply(self._init_weights)
        self._looprec_backbone_adapter.after_owner_init_weights()

        logging.info(
            "LoopRec configuration: backbone=%s, recursion=%s, "
            "router_input=%s, kv_strategy=%s, full_seq_loss=%s",
            self.looprec_backbone,
            self.num_recursion,
            self.looprec_router_input_source,
            self.looprec_kv_cache_strategy,
            bool(self.use_full_seq_loss),
        )

    class Dataset(SequentialModel.Dataset):
        pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
            module.bias.data.zero_()

    def _define_params(self):
        self._looprec_backbone_adapter = build_looprec_backbone_adapter(self, self.looprec_backbone)
        self._looprec_backbone_adapter.attach_backbone_modules()

    def set_looprec_warmup_steps(self, steps_per_epoch: int):
        if self.looprec_warmup_epochs > 0 and self.looprec_warmup_steps == 0:
            computed = int(self.looprec_warmup_epochs) * int(steps_per_epoch)
            self.looprec_warmup_steps = computed
            self.looprec_wrapper.set_warmup_steps(computed)
            logging.info(
                "LoopRec warmup: %s epochs = %s steps (%s steps/epoch)",
                self.looprec_warmup_epochs,
                computed,
                steps_per_epoch,
            )

    def forward(
        self,
        feed_dict: dict,
        epoch: int = 0,
        stage: str = "train",
        return_routing_info: Optional[bool] = None,
    ) -> dict:
        del epoch
        backbone_inputs = self._looprec_backbone_adapter.build_looprec_inputs(feed_dict)
        if return_routing_info is None:
            return_info = (stage != "train") and self._looprec_return_routing_info_in_infer
        else:
            return_info = bool(return_routing_info)

        looprec_out: LoopRecOutput = self.looprec_wrapper(
            backbone_inputs.input_embs,
            backbone_inputs.item_seq_ids,
            self.item_num - 1,
            return_routing_info=return_info,
            return_router_logits=False,
        )
        task_outputs = self._looprec_backbone_adapter.project_looprec_outputs(
            looprec_out.hidden_states,
            backbone_inputs,
            feed_dict,
            stage,
        )

        out = {
            "looprec_output": looprec_out,
            ITEM_ID: feed_dict[ITEM_ID],
            "prediction": task_outputs.prediction,
        }
        out.update(task_outputs.additional_outputs)
        return out

    def loss(self, out_dict: dict) -> torch.Tensor:
        task_loss = self._looprec_backbone_adapter.compute_task_loss(out_dict)
        looprec_out: LoopRecOutput = out_dict["looprec_output"]

        total_loss = task_loss
        if looprec_out.balancing_loss is not None:
            total_loss = total_loss + self.looprec_balancing_coeff * looprec_out.balancing_loss
        if looprec_out.router_z_loss is not None:
            total_loss = total_loss + self.looprec_z_loss_coeff * looprec_out.router_z_loss
        return total_loss
