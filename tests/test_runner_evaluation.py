from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.BaseRunner import BaseRunner
from utils.constants import ITEM_ID


def test_evaluate_method_matches_top_index_path():
    predictions = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.0],
            [0.7, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.8, 0.6],
            [0.0, 0.3, 0.4, 0.5],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0, 2, 3], dtype=torch.long)
    topk = [1, 2, 3]
    metrics = ["HR", "NDCG", "MRR"]

    full_score_results = BaseRunner.evaluate_method(predictions, targets, topk, metrics)
    top_idx = BaseRunner._topk_indices_from_predictions(predictions, topk)
    top_index_results = BaseRunner.evaluate_method_from_top_indices(
        top_idx, targets, topk, metrics
    )

    assert full_score_results == top_index_results


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, feed_dict, stage="infer"):
        return {"prediction": feed_dict["raw_prediction"]}


class _DummyDataset:
    def __init__(self):
        self.model = _DummyModel()
        self.rows = [
            {
                "raw_prediction": torch.tensor([0.1, 0.2, 0.9], dtype=torch.float32),
                ITEM_ID: torch.tensor(1, dtype=torch.long),
            },
            {
                "raw_prediction": torch.tensor([0.3, 0.8, 0.95], dtype=torch.float32),
                ITEM_ID: torch.tensor(1, dtype=torch.long),
            },
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    @staticmethod
    def collate_batch(feed_dicts):
        return {
            "raw_prediction": torch.stack([d["raw_prediction"] for d in feed_dicts]),
            ITEM_ID: torch.stack([d[ITEM_ID] for d in feed_dicts]),
        }


def test_predict_topk_masks_padding_item_before_ranking():
    runner = BaseRunner.__new__(BaseRunner)
    runner.eval_batch_size = 2
    runner.num_workers = 0
    runner.pin_memory = 0

    top_idx, item_ids = runner.predict_topk(_DummyDataset(), [1, 2])

    assert top_idx.tolist() == [[1, 0], [1, 0]]
    assert item_ids.tolist() == [1, 1]


class _LoopRecScheduleDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def set_looprec_warmup_steps(self, steps_per_epoch: int):
        self.warmup_steps = steps_per_epoch


class _LenOnlyDataset:
    def __init__(self, model: nn.Module, length: int):
        self.model = model
        self.length = int(length)

    def __len__(self):
        return self.length


def test_train_preamble_sets_looprec_warmup_steps():
    runner = BaseRunner.__new__(BaseRunner)
    runner.batch_size = 4
    runner.epoch = 1
    runner.warmup_epochs = 0
    runner.warmup_steps = 0
    runner.is_main_process = False
    runner.time = None
    runner.fit = lambda dataset, epoch: float("nan")

    model = _LoopRecScheduleDummyModel()
    train_dataset = _LenOnlyDataset(model, length=10)

    runner.train({"train": train_dataset})

    assert runner.total_steps == 3
    assert model.warmup_steps == 3
