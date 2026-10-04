from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
UPSTREAM_ROOT = REPO_ROOT / "external_assets" / "backbone_sources" / "repos" / "LRURec"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.LRURec import LRURec
from utils.constants import ITEM_ID, ITEM_SEQ, ITEM_SEQ_LEN, MAX_ITEM_SEQ_LEN


UPSTREAM_AVAILABLE = (UPSTREAM_ROOT / "model" / "lru.py").exists()
if UPSTREAM_AVAILABLE:
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.append(str(UPSTREAM_ROOT))
    from model.lru import LRU as UpstreamLRU


class FakeCorpus:
    n_users = 10
    n_items = 30


def make_local_args(**overrides):
    base = dict(
        device=torch.device("cpu"),
        model_path="",
        buffer=0,
        test_all=1,
        dataset="Software",
        data_profile="amazon_abs_v3_rearec",
        emb_size=8,
        num_layers=2,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_full_seq_loss=0,
        lrurec_r_min=0.8,
        lrurec_r_max=0.99,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_upstream_args(**overrides):
    base = dict(
        num_items=FakeCorpus.n_items - 1,
        bert_hidden_units=8,
        bert_num_blocks=2,
        bert_dropout=0.0,
        bert_attn_dropout=0.0,
        dataset_code="software",
        negative_sample_size=100,
        xlong_negative_sample_size=10000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _left_pad_to_max_seq(seq, local_pad_idx):
    return [local_pad_idx] * (MAX_ITEM_SEQ_LEN - len(seq)) + list(seq)


def make_feed(local_pad_idx):
    return {
        ITEM_SEQ: torch.tensor(
            [
                _left_pad_to_max_seq([0, 1, 2, 3, 4, 5], local_pad_idx),
                _left_pad_to_max_seq([2, 3, 4, 5, 6, 7], local_pad_idx),
            ],
            dtype=torch.long,
        ),
        ITEM_ID: torch.tensor([0, 8], dtype=torch.long),
        ITEM_SEQ_LEN: torch.tensor([6, 6], dtype=torch.long),
    }


def build_upstream_and_local_pair(**local_overrides):
    if not UPSTREAM_AVAILABLE:
        pytest.skip("external upstream LRURec source is not available in this worktree")
    upstream = UpstreamLRU(make_upstream_args())
    local = LRURec(make_local_args(**local_overrides), FakeCorpus())
    local.load_state_dict(upstream_to_local_state_dict(upstream.state_dict()), strict=True)
    return upstream.eval(), local.eval()


def upstream_to_local_state_dict(upstream_state):
    mapped = {}
    name_map = {
        "embedding.token.": "item_embeddings.",
        "embedding.layer_norm.": "input_layer_norm.",
        "model.bias": "output_bias",
        "model.lru_blocks.": "lru_blocks.",
    }
    for key, value in upstream_state.items():
        mapped_key = key
        for old, new in name_map.items():
            if old.endswith(".") and mapped_key.startswith(old):
                mapped_key = new + mapped_key[len(old) :]
                break
            if mapped_key == old:
                mapped_key = new
                break
        mapped[mapped_key] = value
    return mapped


def test_lrurec_uses_shifted_internal_ids_so_local_item0_is_valid():
    model = LRURec(make_local_args(), FakeCorpus())
    pad = model.local_padding_idx
    item_seq = torch.tensor([[pad, pad, 0, 1, 2, 3]], dtype=torch.long)

    internal = model._local_to_internal_item_ids(item_seq)

    assert internal.tolist() == [[0, 0, 1, 2, 3, 4]]
    assert torch.count_nonzero(model.item_embeddings.weight[0]) > 0
    assert torch.count_nonzero(model.item_embeddings.weight[1]) > 0


def test_lrurec_native_port_upstream_sequence_and_logit_parity():
    upstream, local = build_upstream_and_local_pair(
        use_full_seq_loss=1,
    )
    feed = make_feed(local.local_padding_idx)
    internal_item_seq_ids = local._local_to_internal_item_ids(feed[ITEM_SEQ])

    with torch.no_grad():
        upstream_scores = upstream(internal_item_seq_ids)[0]
        local_out = local(feed, stage="train")

    torch.testing.assert_close(
        upstream_scores,
        local_out["prediction_internal"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        upstream_scores[..., 1:],
        local_out["prediction"][..., :-1],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        upstream_scores[..., :1],
        local_out["prediction"][..., -1:],
        atol=1e-6,
        rtol=1e-6,
    )


def test_lrurec_full_seq_loss_matches_internal_ce():
    _, local = build_upstream_and_local_pair(
        use_full_seq_loss=1,
    )
    feed = make_feed(local.local_padding_idx)

    with torch.no_grad():
        out = local(feed, stage="train")
        local_loss = local.loss(out)

        labels = torch.cat([feed[ITEM_SEQ][:, 1:], feed[ITEM_ID].unsqueeze(1)], dim=1)
        padding_mask = feed[ITEM_SEQ] != local.local_padding_idx
        valid_mask = torch.cat(
            [
                padding_mask[:, 1:],
                feed[ITEM_ID].ne(local.local_padding_idx).unsqueeze(1),
            ],
            dim=1,
        )
        labels_internal = local._local_targets_to_internal(labels)
        labels_internal[~valid_mask] = local.internal_padding_idx
        logits = out["prediction_internal"]
        expected_loss_flat = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels_internal.view(-1),
            reduction="none",
        )
        expected_loss = (
            expected_loss_flat * valid_mask.view(-1).float()
        ).sum() / valid_mask.sum()

    torch.testing.assert_close(local_loss, expected_loss, atol=1e-7, rtol=0.0)


def test_lrurec_native_last_position_loss_maps_local_targets_to_internal_ids():
    _, local = build_upstream_and_local_pair()
    feed = make_feed(local.local_padding_idx)
    feed[ITEM_ID] = torch.tensor([0, 8], dtype=torch.long)

    with torch.no_grad():
        out = local(feed, stage="train")
        local_loss = local.loss(out)
        labels_internal = local._local_targets_to_internal(feed[ITEM_ID])
        expected_loss = torch.nn.functional.cross_entropy(
            out["prediction_internal"],
            labels_internal,
        )

    assert labels_internal.tolist() == [1, 9]
    torch.testing.assert_close(local_loss, expected_loss, atol=1e-7, rtol=0.0)


def test_lrurec_native_full_seq_loss_smoke():
    model = LRURec(
        make_local_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)

    model.train()
    out = model(feed, stage="train")
    loss = model.loss(out)

    assert out["prediction"].shape == (2, MAX_ITEM_SEQ_LEN, FakeCorpus.n_items)
    assert out["prediction_internal"].shape == (2, MAX_ITEM_SEQ_LEN, FakeCorpus.n_items)
    assert out["padding_mask"].shape == (2, MAX_ITEM_SEQ_LEN)
    assert torch.isfinite(loss)


def test_lrurec_complex_forward_backward_adamw_grad_clip_smoke():
    model = LRURec(
        make_local_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    model.train()
    out = model(feed, stage="train")
    loss = model.loss(out)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)
    assert any(param.is_complex() for param in model.parameters())


def test_lrurec_causal_hidden_states_do_not_depend_on_future_tokens():
    model = LRURec(
        make_local_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    model.eval()
    feed = make_feed(model.local_padding_idx)
    changed_feed = {key: value.clone() for key, value in feed.items()}

    checked_until = MAX_ITEM_SEQ_LEN - 4
    changed_feed[ITEM_SEQ][:, checked_until + 1 :] = torch.tensor(
        [[10, 11, 12], [13, 14, 15]],
        dtype=torch.long,
    )

    with torch.no_grad():
        original = model(feed, stage="train")["sequence_output"]
        changed = model(changed_feed, stage="train")["sequence_output"]

    torch.testing.assert_close(
        original[:, : checked_until + 1, :],
        changed[:, : checked_until + 1, :],
        atol=1e-6,
        rtol=0.0,
    )
