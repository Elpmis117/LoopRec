from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.BaseReader import BaseReader
from utils.constants import ITEM_ID, ITEM_SEQ, ITEM_SEQ_LEN, USER_ID
from utils.corpus_cache import build_corpus_cache_path


def _space_join(values) -> str:
    return " ".join(str(v) for v in values)


def _build_prefix_rows(user_id: int, sequence: list[int]) -> list[dict]:
    rows = []
    for end in range(1, len(sequence)):
        rows.append(
            {
                USER_ID: user_id,
                ITEM_SEQ: _space_join(sequence[:end]),
                ITEM_ID: sequence[end],
            }
        )
    return rows


def _write_split(base_dir: Path, dataset: str, split: str, rows: list[dict]) -> None:
    split_path = base_dir / dataset / f"{dataset}.{split}.remap.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(split_path, index=False)


def _make_reader_args(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=str(path),
        dataset="Software",
        sep=",",
        data_profile="",
    )


def _strip_left_pad(seq_tensor: torch.Tensor, pad_value: int) -> list[int]:
    return [int(x) for x in seq_tensor[seq_tensor != pad_value].tolist()]


def test_base_reader_builds_paper_train_tail_windows(tmp_path: Path):
    root = tmp_path / "processed_clean"
    user1_train = list(range(1, 56))
    user2_train = [10, 11, 12, 13]

    train_rows = _build_prefix_rows(1, user1_train) + _build_prefix_rows(2, user2_train)
    valid_rows = [
        {USER_ID: 1, ITEM_SEQ: _space_join(user1_train), ITEM_ID: 56},
        {USER_ID: 2, ITEM_SEQ: _space_join(user2_train), ITEM_ID: 14},
    ]
    test_rows = [
        {USER_ID: 1, ITEM_SEQ: _space_join(user1_train + [56]), ITEM_ID: 57},
        {USER_ID: 2, ITEM_SEQ: _space_join(user2_train + [14]), ITEM_ID: 15},
    ]

    _write_split(root, "Software", "train", train_rows)
    _write_split(root, "Software", "valid", valid_rows)
    _write_split(root, "Software", "test", test_rows)

    reader = BaseReader(_make_reader_args(root))

    assert len(reader.data_dict["train"][ITEM_ID]) == 2
    assert len(reader.data_dict["valid"][ITEM_ID]) == 2
    assert len(reader.data_dict["test"][ITEM_ID]) == 2

    pad_value = reader.n_items - 1
    train_seq = reader.data_dict["train"][ITEM_SEQ]
    train_target = reader.data_dict["train"][ITEM_ID]
    train_len = reader.data_dict["train"][ITEM_SEQ_LEN]

    assert _strip_left_pad(train_seq[0], pad_value) == list(range(5, 55))
    assert int(train_target[0]) == 55
    assert int(train_len[0]) == 50

    assert _strip_left_pad(train_seq[1], pad_value) == [10, 11, 12]
    assert int(train_target[1]) == 13
    assert int(train_len[1]) == 3


def test_corpus_cache_path_uses_paper_protocol_tag(tmp_path: Path):
    cache_path = build_corpus_cache_path(tmp_path, "Software", "BaseReader")

    assert cache_path.endswith("Software/BaseReader__paper_tail_window_fullseq.pkl")
