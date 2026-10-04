# -*- coding: UTF-8 -*-
"""Ordered-exit routing and shared-block recursion for LoopRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRouter(nn.Module):
    """Single linear projection router."""

    def __init__(self, hidden_size: int, out_dim: int, initializer_range: float = 0.02):
        super().__init__()
        self.router = nn.Linear(hidden_size, out_dim, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)


class MLPRouter(nn.Module):
    """Two-layer MLP router."""

    def __init__(self, hidden_size: int, out_dim: int, initializer_range: float = 0.02):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 2, out_dim, bias=False),
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)


class LocalOrderedExitRouter(nn.Module):
    """Binary continue/exit head shared across recursion depths."""

    def __init__(
        self,
        hidden_size: int,
        num_recursion: int,
        router_type: str,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.layer_emb = nn.Embedding(max(1, num_recursion), hidden_size)
        if router_type == "linear":
            self.router = LinearRouter(hidden_size, 1, initializer_range)
        elif router_type == "mlp":
            self.router = MLPRouter(hidden_size, 1, initializer_range)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=initializer_range)

    def compute_exit_logits(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        layer_ids = torch.full(
            hidden_states.shape[:2],
            layer_idx,
            dtype=torch.long,
            device=hidden_states.device,
        )
        continue_logits = self.router(hidden_states + self.layer_emb(layer_ids)).squeeze(-1)
        logits = torch.stack([torch.zeros_like(continue_logits), continue_logits], dim=-1)
        return torch.where(active_mask.unsqueeze(-1), logits, torch.zeros_like(logits))


def compute_balancing_loss(
    router_probs: torch.Tensor,
    routing_indices: torch.Tensor,
    num_recursion: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Switch-style depth balancing over valid item positions."""

    total_tokens = valid_mask.sum().float().clamp(min=1)
    prob_fraction = (router_probs * valid_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
    prob_fraction = prob_fraction / total_tokens
    load_fraction = torch.bincount(
        routing_indices[valid_mask].reshape(-1),
        minlength=num_recursion,
    ).float()
    load_fraction = load_fraction / total_tokens
    return (prob_fraction * load_fraction * num_recursion).sum()


def compute_router_z_loss(
    router_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Router z-loss over active binary exit decisions."""

    per_decision = torch.square(torch.logsumexp(router_logits, dim=-1))
    denom = valid_mask.sum().float().clamp(min=1)
    return (per_decision * valid_mask.float()).sum() / denom


@dataclass
class LoopRecOutput:
    hidden_states: torch.Tensor
    routing_indices: torch.Tensor
    router_probs: Optional[torch.Tensor] = None
    router_logits: Optional[torch.Tensor] = None
    balancing_loss: Optional[torch.Tensor] = None
    router_z_loss: Optional[torch.Tensor] = None
    depth_counts: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None
    in_warmup: bool = False
    layer_exit_probs: Optional[torch.Tensor] = None
    layer_exit_logits: Optional[torch.Tensor] = None
    layer_exit_masks: Optional[torch.Tensor] = None


class LoopRecWrapper(nn.Module):
    """Ordered-exit looped recursion over a shared recommendation block."""

    def __init__(
        self,
        transformer_layers: nn.ModuleList,
        hidden_size: int,
        num_recursion: int,
        router_type: str = "mlp",
        router_temp: float = 1.0,
        balancing: str = "loss",
        use_z_loss: bool = True,
        warmup_steps: int = 0,
        kv_cache_strategy: str = "recursion_wise",
        use_pre_norm: bool = True,
        dropout_prob: float = 0.5,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        if num_recursion < 1:
            raise ValueError("num_recursion must be positive.")
        if router_temp <= 0.0:
            raise ValueError("router_temp must be positive.")
        if balancing not in {"loss", "none"}:
            raise ValueError("balancing must be one of {'loss', 'none'}.")
        if kv_cache_strategy not in {"recursion_wise", "kv_sharing"}:
            raise ValueError(
                "kv_cache_strategy must be one of "
                "{'recursion_wise', 'kv_sharing'}."
            )

        self.recursive_layers = transformer_layers
        self.hidden_size = hidden_size
        self.num_recursion = int(num_recursion)
        self.router_temp = float(router_temp)
        self.balancing = balancing
        self.use_z_loss = bool(use_z_loss)
        self.warmup_steps = int(warmup_steps)
        self.kv_cache_strategy = kv_cache_strategy
        self.use_pre_norm = bool(use_pre_norm)
        self.training_step = 0

        self.input_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout_prob)
        if self.use_pre_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.router = LocalOrderedExitRouter(
            hidden_size=hidden_size,
            num_recursion=self.num_recursion,
            router_type=router_type,
            initializer_range=initializer_range,
        )

    def set_warmup_steps(self, steps: int):
        self.warmup_steps = int(steps)

    @staticmethod
    def _make_causal_mask(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.expand(batch_size, -1, -1, -1)
        if padding_mask is not None:
            mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    @staticmethod
    def _make_selected_causal_mask(
        selected_positions: torch.Tensor,
        selected_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, selected_len = selected_positions.shape
        mask = torch.full(
            (batch_size, 1, selected_len, selected_len),
            -1e10,
            device=selected_positions.device,
        )
        causal = selected_positions.unsqueeze(-1) >= selected_positions.unsqueeze(-2)
        valid_pair = selected_valid_mask.unsqueeze(-1) & selected_valid_mask.unsqueeze(-2)
        return mask.masked_fill((causal & valid_pair).unsqueeze(1), 0.0)

    def _run_shared_block(
        self,
        layer_stack: nn.ModuleList,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = hidden_states
        for layer in layer_stack:
            output, _ = layer(output, attention_mask, kv_cache=None)
        return output

    def _run_full_sequence_block(
        self,
        hidden_states: torch.Tensor,
        padding_mask_4d: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        attention_mask = self._make_causal_mask(
            batch_size,
            seq_len,
            hidden_states.device,
            padding_mask_4d,
        )
        output = self._run_shared_block(self.recursive_layers, hidden_states, attention_mask)
        if valid_mask is not None:
            output = torch.where(valid_mask.unsqueeze(-1), output, hidden_states)
        return output

    def _run_recursion_wise_block(
        self,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        active_counts = active_mask.sum(dim=1)
        max_active = int(active_counts.max().item())
        if max_active == 0:
            return hidden_states

        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len).masked_fill(~active_mask, seq_len)
        selected_pos = positions.sort(dim=1).values[:, :max_active]
        selected_valid = (
            torch.arange(max_active, device=hidden_states.device).unsqueeze(0)
            < active_counts.unsqueeze(1)
        )
        selected_pos = selected_pos.masked_fill(~selected_valid, 0)

        gathered = hidden_states.gather(
            dim=1,
            index=selected_pos.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        attention_mask = self._make_selected_causal_mask(selected_pos, selected_valid)
        gathered = self._run_shared_block(self.recursive_layers, gathered, attention_mask)

        output = hidden_states.clone()
        batch_idx = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        batch_idx = batch_idx.expand_as(selected_pos)
        output[batch_idx[selected_valid], selected_pos[selected_valid]] = gathered[
            selected_valid
        ]
        return output

    def _run_active_tokens_block(
        self,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor,
        padding_mask_4d: torch.Tensor,
    ) -> torch.Tensor:
        if self.kv_cache_strategy == "recursion_wise":
            return self._run_recursion_wise_block(hidden_states, active_mask)

        if not getattr(self, "_kv_sharing_warned", False):
            warnings.warn(
                "kv_sharing recomputes the full sequence at each recursion depth; "
                "it is used for non-attention backbones that cannot pack active tokens.",
                stacklevel=2,
            )
            self._kv_sharing_warned = True
        return self._run_full_sequence_block(hidden_states, padding_mask_4d)

    @staticmethod
    def _apply_exit_output(
        final_h: torch.Tensor,
        processed_h: torch.Tensor,
        weights: torch.Tensor,
        h_input: torch.Tensor,
        exiting: torch.Tensor,
    ) -> None:
        if not exiting.any():
            return
        exit_output = processed_h * weights.unsqueeze(-1) + h_input
        final_h[exiting] = exit_output[exiting]

    def _finalize_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            return self.final_layer_norm(hidden_states)
        return hidden_states

    def _depth_counts(
        self,
        routing_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.bincount(
            routing_indices[valid_mask].reshape(-1),
            minlength=self.num_recursion,
        ).float()

    def forward(
        self,
        input_embs: torch.Tensor,
        item_seq_ids: torch.Tensor,
        padding_idx: int,
        return_routing_info: bool = False,
        return_router_logits: bool = False,
    ) -> LoopRecOutput:
        if self.training:
            self.training_step += 1

        batch_size, seq_len, _ = input_embs.shape
        device = input_embs.device
        padding_mask_bool = item_seq_ids == padding_idx
        padding_mask_4d = padding_mask_bool.unsqueeze(1).unsqueeze(2)
        valid_mask = ~padding_mask_bool
        in_warmup = self.training and self.training_step <= self.warmup_steps

        h = self.input_layer_norm(input_embs)
        h = self.input_dropout(h)
        h_input = h.clone()
        current_h = self._run_full_sequence_block(
            h,
            padding_mask_4d,
            valid_mask=valid_mask,
        )

        pad_depth_probs = F.one_hot(
            torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),
            num_classes=self.num_recursion,
        ).to(dtype=h.dtype)

        if self.num_recursion == 1:
            depth_probs = torch.where(
                valid_mask.unsqueeze(-1),
                torch.ones(batch_size, seq_len, 1, dtype=h.dtype, device=device),
                pad_depth_probs,
            )
            final_h = h_input.clone()
            self._apply_exit_output(
                final_h,
                current_h,
                torch.ones(batch_size, seq_len, dtype=h.dtype, device=device),
                h_input,
                valid_mask,
            )
            routing_indices = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            return LoopRecOutput(
                hidden_states=self._finalize_hidden(final_h),
                routing_indices=routing_indices.detach(),
                router_probs=depth_probs.detach() if return_routing_info else None,
                router_logits=(
                    depth_probs.clamp_min(1e-12).log()
                    if return_router_logits
                    else (
                        depth_probs.clamp_min(1e-12).log().detach()
                        if return_routing_info
                        else None
                    )
                ),
                depth_counts=self._depth_counts(routing_indices, valid_mask),
                valid_mask=valid_mask,
                in_warmup=in_warmup,
            )

        final_h = h_input.clone()
        routing_indices = torch.full(
            (batch_size, seq_len),
            self.num_recursion - 1,
            dtype=torch.long,
            device=device,
        ).masked_fill(padding_mask_bool, 0)
        depth_probs = torch.zeros(
            batch_size,
            seq_len,
            self.num_recursion,
            dtype=h.dtype,
            device=device,
        )
        layer_exit_logits = h.new_zeros(batch_size, seq_len, self.num_recursion - 1, 2)
        layer_exit_probs = h.new_zeros(batch_size, seq_len, self.num_recursion - 1, 2)
        layer_exit_masks = torch.zeros(
            batch_size,
            seq_len,
            self.num_recursion - 1,
            dtype=torch.bool,
            device=device,
        )

        if in_warmup:
            for _ in range(1, self.num_recursion):
                current_h = self._run_full_sequence_block(
                    current_h,
                    padding_mask_4d,
                    valid_mask=valid_mask,
                )
            depth_probs[:, :, self.num_recursion - 1] = valid_mask.float()
            self._apply_exit_output(
                final_h,
                current_h,
                torch.ones(batch_size, seq_len, dtype=h.dtype, device=device),
                h_input,
                valid_mask,
            )
            depth_probs = torch.where(valid_mask.unsqueeze(-1), depth_probs, pad_depth_probs)
            class_router_logits = depth_probs.clamp_min(1e-12).log()
            return LoopRecOutput(
                hidden_states=self._finalize_hidden(final_h),
                routing_indices=routing_indices.detach(),
                router_probs=depth_probs.detach() if return_routing_info else None,
                router_logits=(
                    class_router_logits
                    if return_router_logits
                    else (class_router_logits.detach() if return_routing_info else None)
                ),
                depth_counts=self._depth_counts(routing_indices, valid_mask),
                valid_mask=valid_mask,
                in_warmup=True,
                layer_exit_probs=(
                    layer_exit_probs.detach() if return_routing_info else None
                ),
                layer_exit_logits=(
                    layer_exit_logits
                    if return_router_logits
                    else (layer_exit_logits.detach() if return_routing_info else None)
                ),
                layer_exit_masks=(
                    layer_exit_masks
                    if return_router_logits
                    else (layer_exit_masks.detach() if return_routing_info else None)
                ),
            )

        continue_mass = valid_mask.float()
        active_mask = valid_mask.clone()
        router_hidden = current_h

        for depth_idx in range(self.num_recursion - 1):
            decision_mask = active_mask
            layer_exit_masks[:, :, depth_idx] = decision_mask
            exit_logits = self.router.compute_exit_logits(
                router_hidden,
                layer_idx=depth_idx,
                active_mask=decision_mask,
            )
            exit_logits = exit_logits / self.router_temp
            exit_probs = F.softmax(exit_logits, dim=-1)
            layer_exit_logits[:, :, depth_idx, :] = exit_logits
            layer_exit_probs[:, :, depth_idx, :] = exit_probs

            raw_exit_mass = continue_mass * exit_probs[..., 0]
            raw_continue_mass = continue_mass * exit_probs[..., 1]
            hard_exit = decision_mask & (exit_probs.argmax(dim=-1) == 0)

            depth_probs[:, :, depth_idx] = torch.where(
                hard_exit,
                continue_mass,
                raw_exit_mass,
            )
            continue_mass = torch.where(
                hard_exit,
                torch.zeros_like(raw_continue_mass),
                raw_continue_mass,
            )
            self._apply_exit_output(final_h, current_h, raw_exit_mass, h_input, hard_exit)
            routing_indices[hard_exit] = depth_idx

            active_mask = decision_mask & ~hard_exit
            if not active_mask.any():
                break

            next_h = self._run_active_tokens_block(current_h, active_mask, padding_mask_4d)
            current_h = torch.where(active_mask.unsqueeze(-1), next_h, current_h)
            router_hidden = current_h

        depth_probs[:, :, self.num_recursion - 1] = continue_mass
        self._apply_exit_output(final_h, current_h, continue_mass, h_input, active_mask)
        depth_probs = torch.where(valid_mask.unsqueeze(-1), depth_probs, pad_depth_probs)
        class_router_logits = depth_probs.clamp_min(1e-12).log()

        balancing_loss = None
        router_z_loss = None
        if self.training:
            if self.balancing == "loss":
                balancing_loss = compute_balancing_loss(
                    depth_probs,
                    routing_indices,
                    self.num_recursion,
                    valid_mask,
                )
            if self.use_z_loss:
                router_z_loss = compute_router_z_loss(
                    layer_exit_logits,
                    layer_exit_masks,
                )

        return LoopRecOutput(
            hidden_states=self._finalize_hidden(final_h),
            routing_indices=routing_indices.detach(),
            router_probs=depth_probs.detach() if return_routing_info else None,
            router_logits=(
                class_router_logits
                if return_router_logits
                else (class_router_logits.detach() if return_routing_info else None)
            ),
            balancing_loss=balancing_loss,
            router_z_loss=router_z_loss,
            depth_counts=self._depth_counts(routing_indices, valid_mask),
            valid_mask=valid_mask,
            in_warmup=False,
            layer_exit_probs=layer_exit_probs.detach() if return_routing_info else None,
            layer_exit_logits=(
                layer_exit_logits
                if return_router_logits
                else (layer_exit_logits.detach() if return_routing_info else None)
            ),
            layer_exit_masks=(
                layer_exit_masks
                if return_router_logits
                else (layer_exit_masks.detach() if return_routing_info else None)
            ),
        )
