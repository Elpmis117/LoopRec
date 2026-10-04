# -*- coding: UTF-8 -*-

import os


PAPER_PROTOCOL_CACHE_TAG = "paper_tail_window_fullseq"


def build_corpus_cache_name(reader_tag: str) -> str:
    return f"{reader_tag}__{PAPER_PROTOCOL_CACHE_TAG}.pkl"


def build_corpus_cache_path(
    data_root,
    dataset: str,
    reader_tag: str = "BaseReader",
) -> str:
    return os.path.join(
        os.fspath(data_root),
        dataset,
        build_corpus_cache_name(reader_tag),
    )
