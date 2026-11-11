"""Loaders for project data. Gold notes get stable internal IDs: gold_000 ... gold_079."""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD_V1 = ROOT / "data_secure" / "gold" / "gold_v1.jsonl"   # source annotations, built by src/data/build_gold.py; for comparison with Miranda et al.
GOLD_V2 = ROOT / "data_secure" / "gold" / "gold_v2.jsonl"   # reviewed annotations, built by src/data/build_gold_v2.py
DEFAULT_GOLD = GOLD_V2                                      # both files are git-ignored
HISTORICAL_GOLD = ROOT / "original_work" / "data" / "gold_standard_80.json"
HISTORICAL_SYNTH = {
    "S1_named": "synthetic_v2_1000.json",
    "S1_name_free": "synthetic_v2_noname_400.json",
    "S2_named": "synthetic_v4_named_more.json",
    "S2_name_free": "synthetic_v4_noname_more.json",
}


@dataclass(frozen=True)
class GoldNote:
    note_id: str
    index: int
    text: str
    entities: tuple
    document_id: str = ""
    extra_occurrences: tuple = ()
    redacted_text: str = ""


def gold_id(index: int) -> str:
    return f"gold_{index:03d}"


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_gold(path=DEFAULT_GOLD) -> list[GoldNote]:
    """Reads the JSONL gold set, or the historical JSON list format."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run: python -m src.data.build_gold, then python -m src.data.build_gold_v2")
    if path.suffix == ".jsonl":
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [GoldNote(r["note_id"], r["index"], r["text"], tuple(r["entities"]), r["document_id"], tuple(r["extra_occurrences"]), r["redacted_text"]) for r in rows]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldNote(gold_id(i), i, r["text"], tuple(r.get("entities", []))) for i, r in enumerate(data)]


def load_historical_synthetic(root=ROOT) -> dict[str, list[dict]]:
    base = pathlib.Path(root) / "original_work" / "data"
    return {k: json.loads((base / v).read_text(encoding="utf-8")) for k, v in HISTORICAL_SYNTH.items()}
