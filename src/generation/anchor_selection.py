"""Fold-specific anchor selection. Anchors can only come from the fold's training partition.

There is no function here that accepts the full gold set, and no module-level anchor list.
"""
from __future__ import annotations

import random

from src.data.annotations import annotations_locatable
from src.data.splits import FoldSpec, LeakageError


def eligible_pool(train_notes) -> list:
    """Training notes whose annotations can all be located, so masking is reliable."""
    return [n for n in train_notes if annotations_locatable(n.text, n.entities)]


def length_strata(pool, k: int = 3) -> list[list]:
    """Split the pool by length into k groups (short, medium, long) using the pool's own ranks."""
    ordered = sorted(pool, key=lambda n: (len(n.text), n.note_id))
    return [ordered[len(ordered) * i // k : len(ordered) * (i + 1) // k] for i in range(k)]


class AnchorSelector:
    """Gives each request one short, one medium and one long note, rotating through each group."""

    def __init__(self, spec: FoldSpec, train_notes, strategy: str = "rotate_length_strata", seed: int = 42, n: int = 3):
        spec.check()
        allowed = set(spec.train_ids)
        if any(note.note_id not in allowed for note in train_notes):
            raise LeakageError(f"fold {spec.fold}: a non-training note was offered as anchor material")
        self.spec, self.strategy, self.n = spec, strategy, n
        pool = eligible_pool(train_notes)
        if len(pool) < n:
            raise ValueError("not enough eligible anchors")
        self.pool_ids = sorted(x.note_id for x in pool)
        self.excluded_ids = sorted(x.note_id for x in train_notes if x.note_id not in set(self.pool_ids))
        rng = random.Random(f"{seed}:{spec.fold}:anchors")
        self.strata = [sorted(group, key=lambda x: x.note_id) for group in length_strata(pool, n)]
        for group in self.strata:
            rng.shuffle(group)

    def for_request(self, request_index: int) -> list:
        if self.strategy == "rotate_length_strata":
            chosen = [group[request_index % len(group)] for group in self.strata]
        elif self.strategy == "fixed_length_quantiles":
            chosen = [sorted(group, key=lambda x: (len(x.text), x.note_id))[len(group) // 2] for group in self.strata]
        else:
            raise ValueError(f"unknown anchor strategy: {self.strategy}")
        if {x.note_id for x in chosen} & set(self.spec.test_ids):
            raise LeakageError(f"fold {self.spec.fold}: test note selected as anchor")
        return chosen
