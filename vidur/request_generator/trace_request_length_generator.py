import logging
from typing import Tuple

import numpy as np
import pandas as pd

from vidur.config import TraceRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)

logger = logging.getLogger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        super().__init__(config)

        self.trace_df = pd.read_csv(config.trace_file)

        prefill_col = config.prefill_column
        decode_col = config.decode_column
        missing_cols = [
            col
            for col in (prefill_col, decode_col)
            if col not in self.trace_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Trace file {config.trace_file} missing columns: {', '.join(missing_cols)}"
            )

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df[prefill_col] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df[decode_col] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        total_tokens_safe = total_tokens.where(total_tokens > 0, 1)
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # deduct the diff tokens from the prefill and decode tokens proportionally
        prefill_tokens_ratio = self.trace_df["num_prefill_tokens"] / total_tokens_safe
        decode_tokens_ratio = self.trace_df["num_decode_tokens"] / total_tokens_safe

        self.trace_df["num_prefill_tokens"] -= (
            np.ceil(diff_tokens * prefill_tokens_ratio)
        ).astype(int)

        self.trace_df["num_decode_tokens"] -= (
            np.ceil(diff_tokens * decode_tokens_ratio)
        ).astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # after clamping to at least 1 token, ensure total does not exceed max_tokens
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        over_limit = (total_tokens - self.config.max_tokens).clip(lower=0)
        if (over_limit > 0).any():
            self.trace_df["num_prefill_tokens"] -= over_limit
            deficit = (1 - self.trace_df["num_prefill_tokens"]).clip(lower=0)
            self.trace_df["num_prefill_tokens"] = self.trace_df[
                "num_prefill_tokens"
            ].clip(lower=1)
            self.trace_df["num_decode_tokens"] -= deficit
            self.trace_df["num_decode_tokens"] = self.trace_df[
                "num_decode_tokens"
            ].clip(lower=1)

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= self.config.max_tokens
        )

        assert all(self.trace_df["num_prefill_tokens"] > 0)

        assert all(self.trace_df["num_decode_tokens"] > 0)

        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.info(
            f"Loaded request length trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        pd_distribution = pd_ratio.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        logger.debug(f"Prompt/decode token ratio stats\n: {pd_distribution}")

        # optionally shuffle the df based on the seed
        if self.config.shuffle:
            self.trace_df = self.trace_df.sample(
                frac=1, random_state=self.config.seed
            ).reset_index(drop=True)
        else:
            self.trace_df = self.trace_df.reset_index(drop=True)
        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return None, None

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["num_prefill_tokens"],
            row["num_decode_tokens"],
        )
