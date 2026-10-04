from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
UPSTREAM_SRC_ROOT = (
    REPO_ROOT / "external_assets" / "backbone_sources" / "repos" / "BSARec" / "src"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from models.BSARec import BSARec
from utils.bsarec_layers import mask_bsarec_internal_padding_logits
from utils.constants import ITEM_ID, ITEM_SEQ, ITEM_SEQ_LEN

UPSTREAM_AVAILABLE = (UPSTREAM_SRC_ROOT / "model" / "bsarec.py").exists()
if UPSTREAM_AVAILABLE:
    if str(UPSTREAM_SRC_ROOT) not in sys.path:
        sys.path.append(str(UPSTREAM_SRC_ROOT))
    from model.bsarec import BSARecModel as UpstreamBSARecModel


class FakeCorpus:
    n_users = 10
    n_items = 9


def make_local_args(**overrides):
    base = dict(
        device=torch.device("cpu"),
        model_path="",
        buffer=0,
        test_all=1,
        dataset="Software",
        data_profile="",
        emb_size=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        use_full_seq_loss=0,
        bsarec_alpha=0.7,
        bsarec_c=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_upstream_args(**overrides):
    base = dict(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        max_seq_length=50,
        batch_size=2,
        alpha=0.7,
        c=5,
        item_size=FakeCorpus.n_items,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_feed(local_pad_idx):
    return {
        ITEM_SEQ: torch.tensor(
            [
                [local_pad_idx, local_pad_idx, 1, 2, 3, 4],
                [local_pad_idx, 2, 3, 4, 5, 6],
            ],
            dtype=torch.long,
        ),
        ITEM_ID: torch.tensor([5, 7], dtype=torch.long),
        ITEM_SEQ_LEN: torch.tensor([4, 5], dtype=torch.long),
    }


def build_upstream_and_local_pair(**local_overrides):
    if not UPSTREAM_AVAILABLE:
        pytest.skip("external upstream BSARec source is not available in this worktree")
    upstream = UpstreamBSARecModel(make_upstream_args())
    local = BSARec(make_local_args(**local_overrides), FakeCorpus())
    local.load_state_dict(upstream.state_dict(), strict=True)
    with torch.no_grad():
        upstream.item_embeddings.weight[0].zero_()
    return upstream.eval(), local.eval()


def test_bsarec_native_port_upstream_sequence_and_logit_parity():
    upstream, local = build_upstream_and_local_pair()
    feed = make_feed(local.local_padding_idx)
    internal_item_seq_ids = local._local_to_internal_item_ids(feed[ITEM_SEQ])

    with torch.no_grad():
        upstream_seq_output = upstream.forward(internal_item_seq_ids)
        local_out = local(feed, stage="test")

    upstream_last = upstream_seq_output[:, -1, :]
    upstream_logits = torch.matmul(upstream_last, upstream.item_embeddings.weight.transpose(0, 1))

    torch.testing.assert_close(
        upstream_seq_output,
        local_out["sequence_output"],
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        upstream_logits,
        local_out["prediction_internal"],
        atol=1e-7,
        rtol=0.0,
    )
    assert local_out["prediction"].shape[-1] == FakeCorpus.n_items


def test_bsarec_native_last_position_loss_maps_local_targets_to_internal_ids():
    _, local = build_upstream_and_local_pair()
    feed = make_feed(local.local_padding_idx)

    with torch.no_grad():
        out = local(feed, stage="train")
        local_loss = local.loss(out)
        labels_internal = local._local_targets_to_internal(feed[ITEM_ID])
        expected_loss = torch.nn.functional.cross_entropy(
            mask_bsarec_internal_padding_logits(out["prediction_internal"]),
            labels_internal,
        )

    torch.testing.assert_close(local_loss, expected_loss, atol=1e-7, rtol=0.0)


def test_bsarec_uses_shifted_internal_ids_so_local_item0_is_valid():
    model = BSARec(make_local_args(), FakeCorpus())
    pad = model.local_padding_idx
    item_seq = torch.tensor([[pad, pad, 0, 1, 2, 3]], dtype=torch.long)

    internal = model._local_to_internal_item_ids(item_seq)
    attention_mask = model._build_attention_mask(internal)

    assert internal.tolist() == [[0, 0, 1, 2, 3, 4]]
    assert torch.count_nonzero(model.item_embeddings.weight[0]) == 0
    assert torch.count_nonzero(model.item_embeddings.weight[1]) > 0
    assert attention_mask[0, 0, -1, 2].item() == 0.0
    assert attention_mask[0, 0, -1, 0].item() < -9999.0


def test_bsarec_target_item0_loss_is_finite_and_uses_internal_row1():
    model = BSARec(make_local_args(), FakeCorpus())
    feed = make_feed(model.local_padding_idx)
    feed[ITEM_ID] = torch.tensor([0, 7], dtype=torch.long)

    out = model(feed, stage="train")
    loss = model.loss(out)
    labels_internal = model._local_targets_to_internal(feed[ITEM_ID])

    assert labels_internal.tolist() == [1, 8]
    assert torch.isfinite(loss)
    expected_loss = torch.nn.functional.cross_entropy(
        mask_bsarec_internal_padding_logits(out["prediction_internal"]),
        labels_internal,
    )
    torch.testing.assert_close(loss, expected_loss, atol=1e-7, rtol=0.0)


def test_bsarec_zeroes_padding_row_after_init_and_before_forward():
    model = BSARec(make_local_args(), FakeCorpus())
    feed = make_feed(model.local_padding_idx)

    torch.testing.assert_close(
        model.item_embeddings.weight[0],
        torch.zeros_like(model.item_embeddings.weight[0]),
    )

    with torch.no_grad():
        model.item_embeddings.weight[0].fill_(7.0)
        model.item_embeddings.weight[1].fill_(3.0)

    model(feed, stage="train")

    torch.testing.assert_close(
        model.item_embeddings.weight[0],
        torch.zeros_like(model.item_embeddings.weight[0]),
    )
    torch.testing.assert_close(
        model.item_embeddings.weight[1],
        torch.full_like(model.item_embeddings.weight[1], 3.0),
    )


def test_bsarec_native_port_full_seq_loss_smoke():
    model = BSARec(
        make_local_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)
    out = model(feed, stage="train")
    loss = model.loss(out)

    assert out["prediction"].shape == (2, 6, FakeCorpus.n_items)
    assert torch.isfinite(loss)
