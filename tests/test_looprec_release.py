from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.LoopRec import LoopRec
from utils.constants import ITEM_ID, ITEM_SEQ, ITEM_SEQ_LEN, MAX_ITEM_SEQ_LEN


class FakeCorpus:
    n_users = 10
    n_items = 30


def make_args(**overrides):
    base = dict(
        device=torch.device("cpu"),
        dataset="Software",
        data_profile="amazon_abs_v3_rearec",
        model_path="",
        buffer=0,
        test_all=1,
        emb_size=8,
        num_layers=2,
        num_heads=2,
        inner_size=16,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        use_full_seq_loss=1,
        looprec_backbone="sasrec",
        num_recursion=2,
        looprec_router_formulation="local_depth_classifier",
        looprec_router_type="mlp",
        looprec_router_func="softmax",
        looprec_router_temp=1.0,
        looprec_router_input_source="first_block_output",
        looprec_localdepth_parametrization="ordered_exit",
        looprec_localdepth_training_mode="selected_only_st",
        looprec_balancing="loss",
        looprec_balancing_coeff=0.01,
        looprec_use_z_loss=1,
        looprec_z_loss_coeff=1e-3,
        looprec_warmup_steps=0,
        looprec_warmup_epochs=0,
        looprec_kv_cache_strategy="recursion_wise",
        looprec_gating="weighted",
        looprec_exited_token_memory="active_only",
        bsarec_alpha=0.9,
        bsarec_c=3,
        nextitnet_dilations="1,2,4,8",
        nextitnet_kernel_size=3,
        nextitnet_block_num=1,
        nextitnet_reg_weight=0.0,
        lrurec_r_min=0.8,
        lrurec_r_max=0.99,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_feed():
    pad = FakeCorpus.n_items - 1

    def left_pad(seq):
        return [pad] * (MAX_ITEM_SEQ_LEN - len(seq)) + list(seq)

    return {
        ITEM_SEQ: torch.tensor(
            [
                left_pad([0, 1, 2, 3, 4, 5]),
                left_pad([2, 3, 4, 5, 6, 7]),
            ],
            dtype=torch.long,
        ),
        ITEM_ID: torch.tensor([6, 8], dtype=torch.long),
        ITEM_SEQ_LEN: torch.tensor([6, 6], dtype=torch.long),
    }


@pytest.mark.parametrize(
    ("backbone", "kv_strategy", "num_recursion"),
    [
        ("sasrec", "recursion_wise", 3),
        ("bsarec", "recursion_wise", 2),
        ("nextitnet", "kv_sharing", 2),
        ("lrurec", "kv_sharing", 2),
    ],
)
def test_looprec_release_mainline_forward_loss_backward(backbone, kv_strategy, num_recursion):
    model = LoopRec(
        make_args(
            looprec_backbone=backbone,
            looprec_kv_cache_strategy=kv_strategy,
            num_recursion=num_recursion,
        ),
        FakeCorpus(),
    )
    feed = make_feed()

    model.train()
    out = model(feed, stage="train")
    loss = model.loss(out)
    loss.backward()

    assert out["prediction"].shape == (2, MAX_ITEM_SEQ_LEN, FakeCorpus.n_items)
    assert out["looprec_output"].routing_indices.shape == (2, MAX_ITEM_SEQ_LEN)
    assert torch.isfinite(loss)
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    )

    model.eval()
    with torch.no_grad():
        infer_out = model(feed, stage="infer", return_routing_info=True)

    looprec_out = infer_out["looprec_output"]
    assert infer_out["prediction"].shape == (2, FakeCorpus.n_items)
    assert looprec_out.router_probs.shape == (2, MAX_ITEM_SEQ_LEN, num_recursion)
    assert looprec_out.depth_counts.sum().item() == 12


def test_looprec_release_warmup_epoch_conversion():
    model = LoopRec(make_args(looprec_warmup_epochs=3), FakeCorpus())

    model.set_looprec_warmup_steps(steps_per_epoch=5)

    assert model.looprec_warmup_steps == 15
    assert model.looprec_wrapper.warmup_steps == 15


def test_looprec_release_rejects_non_mainline_router_input():
    with pytest.raises(ValueError, match="first_block_output"):
        LoopRec(make_args(looprec_router_input_source="input_emb"), FakeCorpus())


@pytest.mark.parametrize(
    ("backbone", "kv_strategy", "message"),
    [
        ("nextitnet", "recursion_wise", "kv_sharing"),
        ("lrurec", "recursion_wise", "kv_sharing"),
        ("sasrec", "kv_sharing", "recursion_wise"),
        ("bsarec", "kv_sharing", "recursion_wise"),
    ],
)
def test_looprec_release_rejects_wrong_kv_strategy(backbone, kv_strategy, message):
    with pytest.raises(ValueError, match=message):
        LoopRec(
            make_args(looprec_backbone=backbone, looprec_kv_cache_strategy=kv_strategy),
            FakeCorpus(),
        )
