#!/usr/bin/env python3
"""
Preprocess the Amazon Reviews 2023 datasets for LoopRec.

Preprocessing consists of two stages:

1. ``amazon_abs_v1``
   Raw JSONL -> rating > 3 -> iterative 5-core -> global timestamp split ->
   history-length and train-item coverage filters -> remapped CSVs.

2. ``amazon_abs_v3_rearec``
   ``amazon_abs_v1`` processed CSVs -> ReaRec-style minimum-history filtering
   with evaluation histories trimmed to the train item vocabulary -> remapped CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = BASE_DIR / "raw"
DEFAULT_CANONICAL_OUTPUT_ROOT = BASE_DIR / "processed_clean"
DEFAULT_CANONICAL_OUTPUT_ROOT_V3_REAREC = BASE_DIR / "processed_clean_v3_rearec"
DEFAULT_CANONICAL_INPUT_ROOT = DEFAULT_CANONICAL_OUTPUT_ROOT

PROFILE_CANONICAL = "amazon_abs_v1"
PROFILE_CANONICAL_V3_REAREC = "amazon_abs_v3_rearec"

DEFAULT_K_CORE = 5
DEFAULT_TRAIN_MIN_HISTORY = 1
DEFAULT_EVAL_MIN_HISTORY = 5
DEFAULT_TRAIN_CUTOFF_RATIO = 0.8
DEFAULT_VALID_CUTOFF_RATIO = 0.9
DEFAULT_REAREC_MIN_HISTORY = 10


DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "Office_Products": {"item_col": "parent_asin"},
    "Patio_Lawn_and_Garden": {"item_col": "parent_asin"},
    "Software": {"item_col": "parent_asin"},
}

StageCounts = List[Tuple[str, int]]
SplitFrames = Dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Amazon Reviews 2023 with absolute-time splits for LoopRec."
        )
    )
    parser.add_argument(
        "--profile",
        choices=[PROFILE_CANONICAL, PROFILE_CANONICAL_V3_REAREC],
        default=PROFILE_CANONICAL,
        help="Data profile to build.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        choices=sorted(DATASET_CONFIG),
        help="Amazon categories to preprocess.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root containing raw JSONL files. Defaults to datasets/raw.",
    )
    parser.add_argument(
        "--canonical-input-root",
        type=Path,
        default=DEFAULT_CANONICAL_INPUT_ROOT,
        help="Root containing amazon_abs_v1 processed CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root for processed outputs. Defaults to datasets/processed_clean "
            "for amazon_abs_v1 and datasets/processed_clean_v3_rearec for "
            "amazon_abs_v3_rearec."
        ),
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=DEFAULT_K_CORE,
        help="Iterative user/item k-core threshold for amazon_abs_v1.",
    )
    parser.add_argument(
        "--train-min-history",
        type=int,
        default=DEFAULT_TRAIN_MIN_HISTORY,
        help="Minimum history length kept in train for amazon_abs_v1.",
    )
    parser.add_argument(
        "--eval-min-history",
        type=int,
        default=DEFAULT_EVAL_MIN_HISTORY,
        help="Minimum history length kept in valid/test for amazon_abs_v1.",
    )
    parser.add_argument(
        "--train-cutoff-ratio",
        type=float,
        default=DEFAULT_TRAIN_CUTOFF_RATIO,
        help="Global timestamp ratio for the end of train in amazon_abs_v1.",
    )
    parser.add_argument(
        "--valid-cutoff-ratio",
        type=float,
        default=DEFAULT_VALID_CUTOFF_RATIO,
        help="Global timestamp ratio for the end of valid in amazon_abs_v1.",
    )
    parser.add_argument(
        "--rearec-min-history",
        type=int,
        default=DEFAULT_REAREC_MIN_HISTORY,
        help="Minimum history length for amazon_abs_v3_rearec rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing processed CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline and print stats without writing output files.",
    )
    parser.add_argument(
        "--skip-remap",
        action="store_true",
        help="Run preprocess only and skip writing *.remap.csv files.",
    )
    parser.add_argument(
        "--remap-only",
        action="store_true",
        help="Skip preprocess and build *.remap.csv from existing processed CSVs.",
    )
    args = parser.parse_args()

    if args.skip_remap and args.remap_only:
        parser.error("--skip-remap and --remap-only cannot be used together.")
    if not (0.0 < args.train_cutoff_ratio < args.valid_cutoff_ratio < 1.0):
        parser.error("Require 0 < train-cutoff-ratio < valid-cutoff-ratio < 1.")
    if args.k_core < 1:
        parser.error("--k-core must be >= 1.")
    if args.train_min_history < 1 or args.eval_min_history < 1:
        parser.error("History thresholds must be >= 1.")
    if args.rearec_min_history < 1:
        parser.error("--rearec-min-history must be >= 1.")

    args.datasets = list(dict.fromkeys(args.dataset))
    if args.output_root is None:
        if args.profile == PROFILE_CANONICAL:
            args.output_root = DEFAULT_CANONICAL_OUTPUT_ROOT
        else:
            args.output_root = DEFAULT_CANONICAL_OUTPUT_ROOT_V3_REAREC
    return args


def read_csv_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def split_history(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return text.split()


def summarize_history(df: pd.DataFrame) -> str:
    if df.empty:
        return "rows=0 min_hist=0 mean_hist=0.00"
    hist_len = df["history"].map(lambda x: len(split_history(x)))
    return (
        f"rows={len(df):,} min_hist={int(hist_len.min())} "
        f"mean_hist={hist_len.mean():.2f}"
    )


def ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite:\n"
            + "\n".join(existing)
        )


def print_stage_counts(split_name: str, stages: StageCounts) -> None:
    print(f"  {split_name}:")
    for stage_name, count in stages:
        print(f"    {stage_name:<24} {count:>10,}")


def extract_item_universe(df: pd.DataFrame, item_col: str) -> Set[str]:
    items: Set[str] = set(df[item_col].dropna().astype(str).unique().tolist())
    for history in df["history"].dropna():
        items.update(split_history(history))
    return items


def get_processed_paths(output_dir: Path, dataset: str) -> Dict[str, Path]:
    return {
        split: output_dir / f"{dataset}.{split}.csv"
        for split in ("train", "valid", "test")
    }


def get_remap_paths(output_dir: Path, dataset: str) -> Dict[str, Path]:
    return {
        split: output_dir / f"{dataset}.{split}.remap.csv"
        for split in ("train", "valid", "test")
    }


def load_processed_splits(output_dir: Path, dataset: str) -> SplitFrames:
    paths = get_processed_paths(output_dir, dataset)
    return {split: read_csv_or_fail(path) for split, path in paths.items()}


def build_remap_maps(
    splits: SplitFrames,
    item_col: str,
) -> Tuple[Dict[object, int], Dict[str, int]]:
    all_users: Set[object] = set()
    all_items: Set[str] = set()
    for df in splits.values():
        all_users.update(df["user_id"].dropna().unique().tolist())
        all_items.update(df[item_col].dropna().astype(str).tolist())
        for history in df["history"].dropna():
            all_items.update(split_history(history))

    user_map = {
        user_id: idx for idx, user_id in enumerate(sorted(all_users, key=lambda x: str(x)))
    }
    item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
    return user_map, item_map


def remap_split_df(
    df: pd.DataFrame,
    item_col: str,
    user_map: Dict[object, int],
    item_map: Dict[str, int],
) -> pd.DataFrame:
    remapped = df.copy()
    remapped["user_id"] = remapped["user_id"].map(user_map)
    remapped["item_id"] = remapped[item_col].map(lambda x: item_map[str(x)])
    remapped["history_id"] = remapped["history"].apply(
        lambda x: " ".join(str(item_map[token]) for token in split_history(x))
    )
    return remapped


def run_remap_stage(
    dataset: str,
    output_dir: Path,
    item_col: str,
    splits: SplitFrames,
    args: argparse.Namespace,
) -> None:
    print("  remap stage:")
    user_map, item_map = build_remap_maps(splits, item_col=item_col)
    print(f"    users                   {len(user_map):>10,}")
    print(f"    items                   {len(item_map):>10,}")

    if args.dry_run:
        print("    dry_run                 skip writing *.remap.csv")
        return

    remap_paths = get_remap_paths(output_dir, dataset)
    ensure_writable(remap_paths.values(), overwrite=args.overwrite)
    remapped_splits = {
        split: remap_split_df(df, item_col=item_col, user_map=user_map, item_map=item_map)
        for split, df in splits.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, df in remapped_splits.items():
        df.to_csv(remap_paths[split], index=False)
        print(f"    wrote                   {remap_paths[split]}")


def load_raw_interactions(raw_jsonl: Path, item_col: str) -> pd.DataFrame:
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Missing raw interaction JSONL: {raw_jsonl}")

    records: List[Dict[str, object]] = []
    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Parsing {raw_jsonl.name}"):
            obj = json.loads(line)
            records.append(
                {
                    "user_id": obj.get("user_id"),
                    item_col: obj.get(item_col),
                    "rating": obj.get("rating"),
                    "timestamp": obj.get("timestamp"),
                }
            )
    return pd.DataFrame(records)


def iterative_k_core(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    k: int,
) -> Tuple[pd.DataFrame, int]:
    if k <= 1:
        return df.reset_index(drop=True), 0

    iterations = 0
    with tqdm(desc=f"K-core filtering (K={k})") as pbar:
        while True:
            user_count = df[user_col].value_counts()
            item_count = df[item_col].value_counts()
            users_to_remove = user_count[user_count < k].index
            items_to_remove = item_count[item_count < k].index
            if len(users_to_remove) + len(items_to_remove) == 0:
                break
            df = df[
                ~df[user_col].isin(users_to_remove)
                & ~df[item_col].isin(items_to_remove)
            ].reset_index(drop=True)
            iterations += 1
            pbar.update(1)
            pbar.set_postfix(
                rows=len(df),
                rm_users=len(users_to_remove),
                rm_items=len(items_to_remove),
            )
    return df, iterations


def build_history_column(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> pd.Series:
    tqdm.pandas(desc="Building user history")
    history = df.groupby(user_col, sort=False, group_keys=False)[item_col].progress_apply(
        lambda x: pd.Series(
            [" ".join(x.values[:i]) for i in range(len(x))],
            index=x.index,
        )
    )
    if isinstance(history.index, pd.MultiIndex):
        history.index = history.index.droplevel(0)
    return history.reindex(df.index)


def timestamp_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_cutoff_ratio: float,
    valid_cutoff_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    min_ts = float(df[timestamp_col].min())
    max_ts = float(df[timestamp_col].max())
    train_cutoff = min_ts + (max_ts - min_ts) * train_cutoff_ratio
    valid_cutoff = min_ts + (max_ts - min_ts) * valid_cutoff_ratio

    train_df = df[df[timestamp_col] <= train_cutoff].copy()
    valid_df = df[
        (df[timestamp_col] > train_cutoff) & (df[timestamp_col] <= valid_cutoff)
    ].copy()
    test_df = df[df[timestamp_col] > valid_cutoff].copy()
    return train_df, valid_df, test_df, {
        "min_ts": min_ts,
        "max_ts": max_ts,
        "train_cutoff": train_cutoff,
        "valid_cutoff": valid_cutoff,
    }


def filter_by_history_threshold(
    df: pd.DataFrame,
    min_history: int,
) -> Tuple[pd.DataFrame, StageCounts]:
    stages: StageCounts = [("input", len(df))]
    cur = df[df["history"].notna()].copy()
    stages.append(("history_notnull", len(cur)))
    cur = cur[cur["history"].map(lambda x: len(split_history(x)) >= min_history)].copy()
    stages.append((f"history_len>={min_history}", len(cur)))
    return cur, stages


def filter_eval_coverage(
    df: pd.DataFrame,
    item_col: str,
    train_items: Set[str],
) -> Tuple[pd.DataFrame, StageCounts]:
    stages: StageCounts = [("input", len(df))]
    cur = df[df[item_col].astype(str).isin(train_items)].copy()
    stages.append(("target_in_train", len(cur)))
    cur = cur[
        cur["history"].map(lambda x: all(token in train_items for token in split_history(x)))
    ].copy()
    stages.append(("history_in_train", len(cur)))
    return cur, stages


def filter_eval_warmstart_trim_history(
    df: pd.DataFrame,
    item_col: str,
    train_items: Set[str],
    min_history: int,
) -> Tuple[pd.DataFrame, StageCounts]:
    stages: StageCounts = [("input", len(df))]
    cur = df[df[item_col].astype(str).isin(train_items)].copy()
    stages.append(("target_in_train", len(cur)))

    def trim_history(value: object) -> str:
        return " ".join(token for token in split_history(value) if token in train_items)

    cur["history"] = cur["history"].map(trim_history)
    cur = cur[cur["history"].map(lambda x: len(split_history(x)) >= min_history)].copy()
    stages.append((f"trimmed_history_len>={min_history}", len(cur)))
    return cur, stages


def print_core_summary(df: pd.DataFrame, item_col: str, label: str) -> None:
    if df.empty:
        print(f"  {label:<18}: empty")
        return
    user_min = int(df["user_id"].value_counts().min())
    item_min = int(df[item_col].astype(str).value_counts().min())
    print(f"  {label:<18}: train_target_user_min={user_min} train_target_item_min={item_min}")


def split_history_len(value: object) -> int:
    return len(split_history(value))


def build_split_summary(df: pd.DataFrame, item_col: str) -> Dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "users": 0,
            "target_items": 0,
            "item_universe": 0,
            "history_mean": 0.0,
            "history_median": 0.0,
            "history_p90": 0.0,
        }

    history_len = df["history"].map(split_history_len)
    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "target_items": int(df[item_col].astype(str).nunique()),
        "item_universe": int(len(extract_item_universe(df, item_col=item_col))),
        "history_mean": float(history_len.mean()),
        "history_median": float(history_len.median()),
        "history_p90": float(history_len.quantile(0.9)),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_processed_outputs(
    dataset: str,
    output_dir: Path,
    splits: SplitFrames,
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_paths = get_processed_paths(output_dir, dataset)
    summary_path = output_dir / "dataset_profile_summary.json"
    ensure_writable(
        list(output_paths.values()) + [summary_path],
        overwrite=args.overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        df.to_csv(output_paths[split], index=False)
    write_json(summary_path, summary)
    print("  preprocess stage   : wrote processed CSVs and audit file")
    for split in ("train", "valid", "test"):
        print(f"    {output_paths[split]}")
    print(f"    {summary_path}")


def preprocess_canonical_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> Tuple[Path, str, SplitFrames]:
    cfg = DATASET_CONFIG[dataset]
    item_col = str(cfg["item_col"])
    raw_jsonl = args.raw_root / dataset / f"{dataset}.jsonl"
    output_dir = args.output_root / dataset

    print("=" * 72)
    print(f"amazon-preprocess-abs.py -- {dataset}")
    print(f"profile       : {PROFILE_CANONICAL}")
    print(f"raw_jsonl     : {raw_jsonl}")
    print(f"output_dir    : {output_dir}")
    print(f"k_core        : {args.k_core}")
    print(f"train_min     : {args.train_min_history}")
    print(f"eval_min      : {args.eval_min_history}")
    print(f"time_cutoffs  : train={args.train_cutoff_ratio:.2f}, valid={args.valid_cutoff_ratio:.2f}")
    print("title_filter  : False")
    print("dedup_rule    : none")

    inter_df = load_raw_interactions(raw_jsonl, item_col=item_col)
    print(f"  raw rows           : {len(inter_df):,}")
    inter_df = inter_df.dropna(subset=["user_id", item_col, "rating", "timestamp"]).copy()
    inter_df[item_col] = inter_df[item_col].astype(str)
    inter_df["timestamp"] = pd.to_numeric(inter_df["timestamp"], errors="coerce")
    inter_df = inter_df.dropna(subset=["timestamp"]).copy()
    inter_df["timestamp"] = inter_df["timestamp"].astype("int64")
    print(f"  non-null rows      : {len(inter_df):,}")

    inter_df = inter_df[inter_df["rating"] > 3].reset_index(drop=True)
    print(f"  rating>3 rows      : {len(inter_df):,}")

    inter_df, k_core_iters = iterative_k_core(
        inter_df,
        user_col="user_id",
        item_col=item_col,
        k=args.k_core,
    )
    print(f"  k_core iterations  : {k_core_iters}")
    print(f"  after k_core rows  : {len(inter_df):,}")
    print(
        "  users/items        : "
        f"{inter_df['user_id'].nunique():,} / {inter_df[item_col].nunique():,}"
    )

    inter_df = inter_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    inter_df["history"] = build_history_column(
        inter_df,
        user_col="user_id",
        item_col=item_col,
    )

    train_df, valid_df, test_df, split_info = timestamp_split(
        inter_df,
        timestamp_col="timestamp",
        train_cutoff_ratio=args.train_cutoff_ratio,
        valid_cutoff_ratio=args.valid_cutoff_ratio,
    )
    print(
        "  timestamp split    : "
        f"min={split_info['min_ts']:.0f} "
        f"train_cutoff={split_info['train_cutoff']:.0f} "
        f"valid_cutoff={split_info['valid_cutoff']:.0f} "
        f"max={split_info['max_ts']:.0f}"
    )
    print(
        "  split rows         : "
        f"train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,}"
    )

    train_df, train_stages = filter_by_history_threshold(
        train_df,
        args.train_min_history,
    )
    valid_df, valid_hist_stages = filter_by_history_threshold(
        valid_df,
        args.eval_min_history,
    )
    test_df, test_hist_stages = filter_by_history_threshold(
        test_df,
        args.eval_min_history,
    )

    train_items = extract_item_universe(train_df, item_col)
    valid_df, valid_cov_stages = filter_eval_coverage(
        valid_df,
        item_col=item_col,
        train_items=train_items,
    )
    test_df, test_cov_stages = filter_eval_coverage(
        test_df,
        item_col=item_col,
        train_items=train_items,
    )

    print_stage_counts("train", train_stages)
    print_stage_counts("valid_history", valid_hist_stages)
    print_stage_counts("valid_coverage", valid_cov_stages)
    print_stage_counts("test_history", test_hist_stages)
    print_stage_counts("test_coverage", test_cov_stages)
    print(f"  train_item_universe : {len(train_items):,}")
    print(f"  train summary       : {summarize_history(train_df)}")
    print(f"  valid summary       : {summarize_history(valid_df)}")
    print(f"  test summary        : {summarize_history(test_df)}")
    print_core_summary(train_df, item_col=item_col, label="final train core")

    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    summary = {
        "profile": PROFILE_CANONICAL,
        "dataset": dataset,
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "params": {
            "k_core": args.k_core,
            "train_min_history": args.train_min_history,
            "eval_min_history": args.eval_min_history,
            "train_cutoff_ratio": args.train_cutoff_ratio,
            "valid_cutoff_ratio": args.valid_cutoff_ratio,
            "rating_filter": "rating > 3",
            "title_filter": False,
            "dedup_rule": "none",
        },
        "split_timestamp": split_info,
        "final_splits": {
            "train": build_split_summary(train_df, item_col=item_col),
            "valid": build_split_summary(valid_df, item_col=item_col),
            "test": build_split_summary(test_df, item_col=item_col),
        },
        "filter_stages": {
            "train": train_stages,
            "valid_history": valid_hist_stages,
            "valid_coverage": valid_cov_stages,
            "test_history": test_hist_stages,
            "test_coverage": test_cov_stages,
        },
    }

    if args.dry_run:
        print("  preprocess stage   : dry_run, skip writing processed CSVs and audit file")
        return output_dir, item_col, splits

    write_processed_outputs(
        dataset=dataset,
        output_dir=output_dir,
        splits=splits,
        summary=summary,
        args=args,
    )
    return output_dir, item_col, splits


def preprocess_canonical_v3_rearec_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> Tuple[Path, str, SplitFrames]:
    cfg = DATASET_CONFIG[dataset]
    item_col = str(cfg["item_col"])
    input_dir = args.canonical_input_root / dataset
    output_dir = args.output_root / dataset
    min_history = int(args.rearec_min_history)

    print("=" * 72)
    print(f"amazon-preprocess-abs.py -- {dataset}")
    print(f"profile         : {PROFILE_CANONICAL_V3_REAREC}")
    print(f"input_dir       : {input_dir}")
    print(f"output_dir      : {output_dir}")
    print(f"min_history     : {min_history}")
    print("closure         : none")
    print("title_filter    : False")
    print("eval_history    : trim_to_train_vocabulary")

    base_train = read_csv_or_fail(input_dir / f"{dataset}.train.csv")
    base_valid = read_csv_or_fail(input_dir / f"{dataset}.valid.csv")
    base_test = read_csv_or_fail(input_dir / f"{dataset}.test.csv")
    for df in (base_train, base_valid, base_test):
        df[item_col] = df[item_col].astype(str)

    print(
        "  base rows          : "
        f"train={len(base_train):,} valid={len(base_valid):,} test={len(base_test):,}"
    )

    train_df, train_stages = filter_by_history_threshold(base_train, min_history)
    train_items = extract_item_universe(train_df, item_col)

    valid_df, valid_hist_stages = filter_by_history_threshold(base_valid, min_history)
    valid_df, valid_cov_stages = filter_eval_warmstart_trim_history(
        valid_df,
        item_col=item_col,
        train_items=train_items,
        min_history=min_history,
    )
    test_df, test_hist_stages = filter_by_history_threshold(base_test, min_history)
    test_df, test_cov_stages = filter_eval_warmstart_trim_history(
        test_df,
        item_col=item_col,
        train_items=train_items,
        min_history=min_history,
    )

    print_stage_counts("train", train_stages)
    print_stage_counts("valid_history", valid_hist_stages)
    print_stage_counts("valid_coverage", valid_cov_stages)
    print_stage_counts("test_history", test_hist_stages)
    print_stage_counts("test_coverage", test_cov_stages)
    print(f"  train_item_universe : {len(train_items):,}")
    print(f"  train summary       : {summarize_history(train_df)}")
    print(f"  valid summary       : {summarize_history(valid_df)}")
    print(f"  test summary        : {summarize_history(test_df)}")
    print_core_summary(train_df, item_col=item_col, label="final train core")

    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    summary = {
        "profile": PROFILE_CANONICAL_V3_REAREC,
        "dataset": dataset,
        "input_root": str(args.canonical_input_root),
        "output_root": str(args.output_root),
        "params": {
            "min_history": min_history,
            "source_profile": PROFILE_CANONICAL,
            "title_filter": False,
            "eval_filter": "target_warmstart_trim_history_to_train_vocabulary",
        },
        "base_splits": {
            "train": build_split_summary(base_train, item_col=item_col),
            "valid": build_split_summary(base_valid, item_col=item_col),
            "test": build_split_summary(base_test, item_col=item_col),
        },
        "final_splits": {
            "train": build_split_summary(train_df, item_col=item_col),
            "valid": build_split_summary(valid_df, item_col=item_col),
            "test": build_split_summary(test_df, item_col=item_col),
        },
        "filter_stages": {
            "train": train_stages,
            "valid_history": valid_hist_stages,
            "valid_coverage": valid_cov_stages,
            "test_history": test_hist_stages,
            "test_coverage": test_cov_stages,
        },
    }

    if args.dry_run:
        print("  preprocess stage   : dry_run, skip writing processed CSVs and audit file")
        return output_dir, item_col, splits

    write_processed_outputs(
        dataset=dataset,
        output_dir=output_dir,
        splits=splits,
        summary=summary,
        args=args,
    )
    return output_dir, item_col, splits


def preprocess_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> Tuple[Path, str, SplitFrames]:
    if args.profile == PROFILE_CANONICAL:
        return preprocess_canonical_dataset(dataset, args)
    return preprocess_canonical_v3_rearec_dataset(dataset, args)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        output_dir = args.output_root / dataset
        item_col = str(DATASET_CONFIG[dataset]["item_col"])

        if args.remap_only:
            print("=" * 72)
            print(f"amazon-preprocess-abs.py -- {dataset}")
            print(f"profile         : {args.profile}")
            print(f"output_dir       : {output_dir}")
            print("mode            : remap_only")
            splits = load_processed_splits(output_dir, dataset)
        else:
            output_dir, item_col, splits = preprocess_dataset(dataset, args)

        if not args.skip_remap:
            run_remap_stage(
                dataset=dataset,
                output_dir=output_dir,
                item_col=item_col,
                splits=splits,
                args=args,
            )


if __name__ == "__main__":
    main()
