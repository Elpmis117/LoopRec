# -*- coding: UTF-8 -*-

import os
from typing import Final


PROFILE_AMAZON_CANONICAL: Final[str] = "amazon_abs_v1"
PROFILE_AMAZON_V3_REAREC: Final[str] = "amazon_abs_v3_rearec"

CANONICAL_ROOT_NAME: Final[str] = "processed_clean"
CANONICAL_V3_REAREC_ROOT_NAME: Final[str] = "processed_clean_v3_rearec"

PAPER_DATASETS: Final[set[str]] = {
    "Software",
    "Office_Products",
    "Patio_Lawn_and_Garden",
}

SUPPORTED_PROFILES: Final[set[str]] = {
    PROFILE_AMAZON_CANONICAL,
    PROFILE_AMAZON_V3_REAREC,
}


def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path or ""))


def real_path(path: str) -> str:
    return os.path.realpath(normalize_path(path))


def root_basename(path: str, resolve_symlinks: bool = False) -> str:
    normalized = real_path(path) if resolve_symlinks else normalize_path(path)
    return os.path.basename(os.path.normpath(normalized))


def infer_data_profile(dataset: str, path: str) -> str:
    if dataset not in PAPER_DATASETS:
        return ""

    root_name = root_basename(path, resolve_symlinks=True)
    if root_name == CANONICAL_ROOT_NAME:
        return PROFILE_AMAZON_CANONICAL
    if root_name == CANONICAL_V3_REAREC_ROOT_NAME:
        return PROFILE_AMAZON_V3_REAREC
    return ""


def validate_dataset_path(path: str, dataset: str, data_profile: str = "") -> str:
    if dataset not in PAPER_DATASETS:
        supported = ", ".join(sorted(PAPER_DATASETS))
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported datasets: {supported}."
        )

    if data_profile and data_profile not in SUPPORTED_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(
            f"Unsupported data profile: {data_profile}. "
            f"Expected one of: {supported}."
        )

    original_root = root_basename(path, resolve_symlinks=False)
    if original_root == "processed":
        raise ValueError(
            "`datasets/processed` is not a supported data root. "
            "Use `datasets/processed_clean_v3_rearec` for training and evaluation "
            "or `datasets/processed_clean` for the first preprocessing stage."
        )

    inferred_profile = infer_data_profile(dataset, path)
    resolved_root = root_basename(path, resolve_symlinks=True)
    if data_profile == PROFILE_AMAZON_V3_REAREC and (
        resolved_root != CANONICAL_V3_REAREC_ROOT_NAME
    ):
        raise ValueError(
            "Path/profile mismatch: amazon_abs_v3_rearec must point to "
            "`datasets/processed_clean_v3_rearec`."
        )
    if data_profile == PROFILE_AMAZON_CANONICAL and resolved_root != CANONICAL_ROOT_NAME:
        raise ValueError(
            "Path/profile mismatch: amazon_abs_v1 must point to "
            "`datasets/processed_clean`."
        )

    return inferred_profile
