"""Reproducible split seeds and PBS worker settings."""

from __future__ import annotations

import os

import numpy as np


def derive_inner_seed(
    split_index,
    *,
    master_seed,
):
    return int(
        np.random.SeedSequence(
            [
                master_seed,
                split_index,
                1,
            ]
        ).generate_state(
            1
        )[0]
    )


def get_worker_count():
    value = os.environ.get(
        "PBS_NCPUS",
        "1",
    )

    try:
        workers = int(value)

    except ValueError:
        workers = 1

    return max(
        1,
        workers,
    )
