# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

from utils import utils
from utils.constants import *
from utils.data_roots import validate_dataset_path


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument(
            "--path",
            type=str,
            default="../../datasets/processed_clean_v3_rearec",
            help="Input data dir.",
        )
        parser.add_argument(
            "--dataset", type=str, default="Software", help="Choose a dataset."
        )
        parser.add_argument("--sep", type=str, default=",", help="sep of csv file.")
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        validate_dataset_path(
            self.prefix,
            self.dataset,
            getattr(args, "data_profile", ""),
        )
        self._read_data()

    @staticmethod
    def _sequence_length(seq) -> int:
        return int(np.asarray(seq).size)

    @staticmethod
    def _to_int_list(seq) -> list[int]:
        arr = np.asarray(seq, dtype=np.int64)
        return [int(x) for x in arr.tolist()]

    def _recover_user_train_sequences(self, train_df: pd.DataFrame) -> dict[int, list[int]]:
        if train_df.empty:
            return {}

        seq_lengths = train_df[ITEM_SEQ].apply(self._sequence_length)
        longest_indices = seq_lengths.groupby(train_df[USER_ID]).idxmax()
        longest_rows = train_df.loc[longest_indices].sort_values(USER_ID)

        user_sequences = dict()
        for _, row in longest_rows.iterrows():
            history = self._to_int_list(row[ITEM_SEQ])
            target = int(row[ITEM_ID])
            user_sequences[int(row[USER_ID])] = history + [target]
        return user_sequences

    def _build_paper_train_df(self, train_df: pd.DataFrame) -> pd.DataFrame:
        user_sequences = self._recover_user_train_sequences(train_df)
        rows = []
        for user_id, sequence in user_sequences.items():
            if len(sequence) < 2:
                continue

            history = sequence[max(0, len(sequence) - 1 - MAX_ITEM_SEQ_LEN): -1]
            target = sequence[-1]
            rows.append(
                {
                    USER_ID: user_id,
                    ITEM_SEQ: np.array(history, dtype=np.int64),
                    ITEM_ID: int(target),
                }
            )

        return pd.DataFrame(rows, columns=[USER_ID, ITEM_SEQ, ITEM_ID])

    def _compute_item_upper_bound(self, data_df: dict[str, pd.DataFrame]) -> int:
        item_max = 0
        history_max = 0
        for key in ["train", "valid", "test"]:
            if data_df[key].empty:
                continue
            item_max = max(item_max, int(data_df[key][ITEM_ID].max()))
            for seq in data_df[key][ITEM_SEQ].values:
                arr = np.asarray(seq)
                if arr.size:
                    history_max = max(history_max, int(arr.max()))
        return max(item_max, history_max)

    def _read_data(self):
        logging.info(
            'Reading data from "{}", dataset = "{}" '.format(self.prefix, self.dataset)
        )
        data_df = dict()
        for key in ["train", "valid", "test"]:
            data_df[key] = pd.read_csv(
                os.path.join(
                    self.prefix, self.dataset, f"{self.dataset}.{key}.remap.csv"
                ),
                sep=self.sep,
            ).reset_index(drop=True)
            data_df[key] = utils.eval_list_columns(data_df[key])

        logging.info("Counting dataset statistics...")
        key_columns = [USER_ID, ITEM_ID, ITEM_SEQ]
        stats_df = pd.concat(
            [data_df[key][key_columns] for key in ["train", "valid", "test"]],
            ignore_index=True,
        )
        self.n_users = stats_df[USER_ID].nunique() + 1  # including [PAD]
        self.n_items = self._compute_item_upper_bound(data_df) + 2  # including [PAD]
        logging.info(
            '"# user": {}, "# item": {}, "# entry": {}'.format(
                self.n_users - 1, self.n_items - 1, len(stats_df)
            )
        )

        original_train_size = len(data_df["train"])
        data_df["train"] = self._build_paper_train_df(data_df["train"])
        logging.info(
            "Rebuilt train split for paper protocol: %s -> %s rows",
            original_train_size,
            len(data_df["train"]),
        )

        all_df = pd.concat(
            [data_df[key][key_columns] for key in ["train", "valid", "test"]],
            ignore_index=True,
        )
        all_df[ITEM_SEQ] = all_df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])
        all_df[ITEM_SEQ_LEN] = all_df[ITEM_SEQ].apply(len)

        all_data_dict = dict()
        all_data_dict[USER_ID] = torch.from_numpy(all_df[USER_ID].values).long()
        all_data_dict[ITEM_ID] = torch.from_numpy(all_df[ITEM_ID].values).long()
        item_seq = [
            torch.from_numpy(np.array(x)).long() for x in all_df[ITEM_SEQ].values
        ]

        # Left Padding
        left_padded_seqs = [
            pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=self.n_items - 1)
            for seq in item_seq
        ]
        all_data_dict[ITEM_SEQ] = torch.stack(left_padded_seqs)

        all_data_dict[ITEM_SEQ_LEN] = torch.from_numpy(
            all_df[ITEM_SEQ_LEN].values
        ).long()

        self.data_dict = {key: dict() for key in ["train", "valid", "test"]}
        start, end = 0, 0
        for key in ["train", "valid", "test"]:
            end = start + len(data_df[key][ITEM_ID])
            for c in all_data_dict:
                self.data_dict[key][c] = all_data_dict[c][start:end]
            start = end

        logging.info(f"size of train: {len(self.data_dict['train'][ITEM_ID])}")
        logging.info(f"size of valid: {len(self.data_dict['valid'][ITEM_ID])}")
        logging.info(f"size of test: {len(self.data_dict['test'][ITEM_ID])}")
        logging.info("Finish reading data.")
