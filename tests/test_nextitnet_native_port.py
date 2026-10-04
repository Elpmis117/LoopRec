from types import SimpleNamespace
import sys
from pathlib import Path
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.NextItNet import NextItNet
from utils.constants import ITEM_ID, ITEM_SEQ, ITEM_SEQ_LEN, MAX_ITEM_SEQ_LEN
from utils.nextitnet_layers import build_dilation_schedule
from utils.nextitnet_layers import NextItNetResidualBlock
from utils.nextitnet_layers import zero_nextitnet_padding_embedding


class FakeCorpus:
    n_users = 10
    n_items = 30


def make_args(**overrides):
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
        initializer_range=0.02,
        use_full_seq_loss=0,
        nextitnet_dilations="1,2,4,8",
        nextitnet_kernel_size=3,
        nextitnet_block_num=1,
        nextitnet_reg_weight=0.0,
        nextitnet_output_head="tied_embedding",
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


def test_nextitnet_dilation_schedule_repeats_and_truncates():
    assert build_dilation_schedule("1,2,4", num_layers=5, block_num=1) == [1, 2, 4, 1, 2]
    assert build_dilation_schedule("1,4", num_layers=5, block_num=2) == [1, 4, 1, 4, 1]


def test_nextitnet_native_last_position_forward_loss_smoke():
    model = NextItNet(make_args(), FakeCorpus())
    feed = make_feed(model.local_padding_idx)

    model.train()
    out = model(feed, stage="train")
    loss = model.loss(out)

    assert out["prediction"].shape == (2, FakeCorpus.n_items)
    assert out["prediction_internal"].shape == (2, FakeCorpus.n_items)
    assert out["sequence_output"].shape == (2, MAX_ITEM_SEQ_LEN, model.hidden_size)
    assert torch.isfinite(loss)


def test_nextitnet_native_full_seq_loss_smoke():
    model = NextItNet(
        make_args(use_full_seq_loss=1),
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


def test_nextitnet_default_output_head_is_tied_embedding():
    model = NextItNet(make_args(), FakeCorpus())

    assert model.nextitnet_output_head == "tied_embedding"
    assert model.output_layer is None


def test_nextitnet_untied_output_head_full_seq_loss_smoke():
    model = NextItNet(
        make_args(
            use_full_seq_loss=1,
            nextitnet_output_head="untied_linear",
        ),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)

    model.train()
    out = model(feed, stage="train")
    loss = model.loss(out)
    loss.backward()

    assert model.output_layer is not None
    assert model.output_layer.weight.shape == (FakeCorpus.n_items, model.hidden_size)
    assert out["prediction"].shape == (2, MAX_ITEM_SEQ_LEN, FakeCorpus.n_items)
    assert torch.isfinite(loss)
    assert model.output_layer.weight.grad is not None


def test_nextitnet_untied_output_head_last_position_eval_smoke():
    model = NextItNet(
        make_args(nextitnet_output_head="untied_linear"),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)

    out = model(feed, stage="test")

    assert out["prediction"].shape == (2, FakeCorpus.n_items)
    assert out["prediction_internal"].shape == (2, FakeCorpus.n_items)


def test_nextitnet_untied_output_head_does_not_reuse_item_embedding_weight():
    model = NextItNet(
        make_args(nextitnet_output_head="untied_linear"),
        FakeCorpus(),
    )

    assert model.output_layer is not None
    assert model.output_layer.weight.data_ptr() != model.item_embeddings.weight.data_ptr()


def test_nextitnet_sequence_output_zeroes_padding_positions():
    model = NextItNet(
        make_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    feed = make_feed(model.local_padding_idx)

    out = model(feed, stage="train")
    padding_mask = feed[ITEM_SEQ] == model.local_padding_idx

    torch.testing.assert_close(
        out["sequence_output"][padding_mask],
        torch.zeros_like(out["sequence_output"][padding_mask]),
    )


def test_nextitnet_residual_block_masks_padding_between_convs():
    block = NextItNetResidualBlock(hidden_size=2, kernel_size=3, dilation=1)
    valid_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
    hidden = torch.zeros(1, 4, 2)
    captured = {}

    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.ln1.weight.fill_(1.0)
        block.ln1.bias.fill_(1.0)

    def capture_conv2_input(_module, inputs):
        captured["conv2_input"] = inputs[0].detach().clone()

    handle = block.conv2.register_forward_pre_hook(capture_conv2_input)
    try:
        out = block(hidden, valid_mask=valid_mask)
    finally:
        handle.remove()

    conv2_input = captured["conv2_input"]
    left_pad = (block.kernel_size - 1) * block.dilation * 2
    original_positions = conv2_input[:, :, :, left_pad : left_pad + valid_mask.size(1)]

    torch.testing.assert_close(
        original_positions.permute(0, 2, 3, 1).squeeze(1)[~valid_mask],
        torch.zeros_like(out[~valid_mask]),
    )
    torch.testing.assert_close(out[~valid_mask], torch.zeros_like(out[~valid_mask]))


def test_nextitnet_uses_local_item_space_and_zeroes_padding_after_init():
    model = NextItNet(make_args(), FakeCorpus())

    assert model.local_padding_idx == FakeCorpus.n_items - 1
    assert model.internal_item_size == FakeCorpus.n_items
    assert model.item_embeddings.num_embeddings == FakeCorpus.n_items
    assert model.item_embeddings.padding_idx == model.local_padding_idx
    assert torch.count_nonzero(model.item_embeddings.weight[0]) > 0
    torch.testing.assert_close(
        model.item_embeddings.weight[model.local_padding_idx],
        torch.zeros_like(model.item_embeddings.weight[model.local_padding_idx]),
    )


def test_nextitnet_padding_zero_helper_repairs_manual_mutation():
    model = NextItNet(make_args(), FakeCorpus())
    with torch.no_grad():
        model.item_embeddings.weight[0].fill_(3.0)
        model.item_embeddings.weight[model.local_padding_idx].fill_(7.0)

    zero_nextitnet_padding_embedding(model.item_embeddings)

    torch.testing.assert_close(
        model.item_embeddings.weight[0],
        torch.full_like(model.item_embeddings.weight[0], 3.0),
    )
    torch.testing.assert_close(
        model.item_embeddings.weight[model.local_padding_idx],
        torch.zeros_like(model.item_embeddings.weight[model.local_padding_idx]),
    )


def test_nextitnet_forward_repairs_padding_embedding_mutation():
    model = NextItNet(make_args(), FakeCorpus())
    feed = make_feed(model.local_padding_idx)

    with torch.no_grad():
        model.item_embeddings.weight[0].fill_(3.0)
        model.item_embeddings.weight[model.local_padding_idx].fill_(7.0)

    model(feed, stage="train")

    torch.testing.assert_close(
        model.item_embeddings.weight[0],
        torch.full_like(model.item_embeddings.weight[0], 3.0),
    )
    torch.testing.assert_close(
        model.item_embeddings.weight[model.local_padding_idx],
        torch.zeros_like(model.item_embeddings.weight[model.local_padding_idx]),
    )

def test_nextitnet_causal_hidden_states_do_not_depend_on_future_tokens():
    model = NextItNet(
        make_args(use_full_seq_loss=1),
        FakeCorpus(),
    )
    model.eval()
    feed = make_feed(model.local_padding_idx)
    changed_feed = {key: value.clone() for key, value in feed.items()}

    # Change only future positions relative to the checked prefix.
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
