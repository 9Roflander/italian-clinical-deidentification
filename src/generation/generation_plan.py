"""Deterministic per-fold generation plan, built from training-partition statistics only.

The plan is shared by both arms, so specialty, gender, age, length and entity counts stay
identical between style-anchored and non-anchored generation.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import random
import re
from dataclasses import asdict, dataclass, replace

from src.data.annotations import canonical_type


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    fold: int
    record_type: str
    order_index: int
    type_order: int
    specialty: str
    gender: str
    age_band: str
    area: str
    name_form: str
    name_letter: str
    n_nome: int
    n_eta: int
    n_data: int
    n_luogo: int
    target_chars: int
    density_level: str
    # v2 only. None in v1 plans, which keeps v1 prompts and plan hashes byte-identical.
    reference_year: int | None = None
    patient_age: int | None = None

    @property
    def counts(self) -> dict:
        return {"NOME": self.n_nome, "ETÀ": self.n_eta, "DATA": self.n_data, "LUOGO/INDIRIZZO": self.n_luogo}


def note_profile(note) -> dict:
    c = collections.Counter(canonical_type(e.get("type")) for e in note.entities)
    return {"n_nome": c["NOME"], "n_eta": c["ETÀ"], "n_data": c["DATA"], "n_luogo": c["LUOGO/INDIRIZZO"],
            "chars": len(note.text), "total": sum(v for k, v in c.items() if k)}


def age_band_weights(train_notes, bands: list[dict]) -> dict:
    """Bands weighted by numeric ages in the training partition, with +1 smoothing."""
    weights = {b["label"]: 1 for b in bands}
    for n in train_notes:
        for e in n.entities:
            if canonical_type(e.get("type")) == "ETÀ" and str(e.get("text", "")).strip().isdigit():
                age = int(e["text"])
                for b in bands:
                    if b["min"] <= age <= b["max"]:
                        weights[b["label"]] += 1
    return weights


YEAR_IN_DATE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def year_range(train_notes, details: dict) -> tuple[int, int, str]:
    """Years found in the training partition's date mentions, or the configured fallback when too few exist."""
    years = sorted({int(m.group(1)) for n in train_notes for e in n.entities if canonical_type(e.get("type")) == "DATA"
                    for m in YEAR_IN_DATE.finditer(str(e.get("text", "")))})
    if len(years) >= details["min_distinct_training_years"]:
        return years[0], years[-1], "training_partition_dates"
    return details["fallback_year_min"], details["fallback_year_max"], "configured_fallback"


def band_limits(band: dict) -> tuple[int, int]:
    """Ages drawn within the band's label, so 66-95 means 66 to 95 even if the counting limit is wider."""
    low, high = (int(x) for x in band["label"].split("-"))
    return max(1, low), high


def density(total: int) -> str:
    return "nessuna" if total == 0 else "bassa" if total <= 2 else "media" if total <= 5 else "alta"


def _valid_specialties(cfg, gender, age_band):
    out = []
    for name, rule in cfg["specialties"].items():
        rule = rule or {}
        if gender in rule.get("exclude_gender", []) or age_band in rule.get("exclude_age", []):
            continue
        if rule.get("only_age") and age_band not in rule["only_age"]:
            continue
        out.append(name)
    return out


def build_plan(fold: int, train_notes, cfg: dict) -> list[CaseSpec]:
    p, v = cfg["plan"], cfg["validation"]
    rng = random.Random(f"{cfg['seed']}:{fold}:plan")
    profiles = [note_profile(n) for n in sorted(train_notes, key=lambda x: x.note_id)]
    eligible = [x for x in profiles if x["total"] >= v["min_entities"]]
    named = [x for x in eligible if x["n_nome"] > 0]
    free = [x for x in eligible if x["n_nome"] == 0]
    if not named or not free:
        raise ValueError(f"fold {fold}: no profiles to sample from")
    weights = age_band_weights(train_notes, p["age_bands"])
    labels, wvals = list(weights), list(weights.values())
    kinds = (["synthetic_named"] * p["n_named"] + ["synthetic_name_free"] * p["n_name_free"]
             + ["synthetic_negative"] * p["n_negative"])
    rng.shuffle(kinds)
    specs, per_type = [], collections.Counter()
    for order, kind in enumerate(kinds):
        per_type[kind] += 1
        band = rng.choices(labels, wvals)[0]
        gender = rng.choice(p["genders"])
        specialty = rng.choice(_valid_specialties(p, gender, band))
        source = named if kind == "synthetic_named" else free if kind == "synthetic_name_free" else profiles
        prof = rng.choice(source)
        counts = (0, 0, 0, 0) if kind == "synthetic_negative" else (prof["n_nome"], prof["n_eta"], prof["n_data"], prof["n_luogo"])
        chars = int(prof["chars"] * rng.uniform(0.9, 1.1))
        chars = max(v["min_chars"], min(v["max_chars"], chars))
        specs.append(CaseSpec(f"F{fold}-N{order:05d}", fold, kind, order, per_type[kind] - 1, specialty, gender, band,
                              rng.choice(p["areas"]), rng.choices(list(p["name_forms"]), list(p["name_forms"].values()))[0],
                              rng.choice(p["name_letters"]), *counts, chars, density(sum(counts))))
    details = p.get("details")
    if not details:
        return specs
    # A separate random stream, so v1 draws above stay identical.
    rng2 = random.Random(f"{cfg['seed']}:{fold}:plan_details")
    low_year, high_year, _ = year_range(train_notes, details)
    bands = {b["label"]: b for b in p["age_bands"]}
    out = []
    for s in specs:
        year = rng2.randint(low_year, high_year) if s.n_data and details.get("reference_year") else None
        age = rng2.randint(*band_limits(bands[s.age_band])) if s.n_eta and details.get("patient_age") else None
        out.append(replace(s, reference_year=year, patient_age=age))
    return out


def plan_details(train_notes, cfg: dict) -> dict:
    details = cfg["plan"].get("details")
    if not details:
        return {}
    low, high, source = year_range(train_notes, details)
    return {"reference_year_range": [low, high], "reference_year_source": source}


V2_FIELDS = ("reference_year", "patient_age")


def _hashable(spec) -> dict:
    """v1 plans hash exactly as before: unset v2 fields are left out."""
    d = asdict(spec)
    for k in V2_FIELDS:
        if d[k] is None:
            del d[k]
    return d


def plan_sha(specs) -> str:
    return hashlib.sha256(json.dumps([_hashable(s) for s in specs], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_plan(specs, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(specs[0]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in specs:
            w.writerow(asdict(s))


def plan_summary(specs) -> dict:
    n = len(specs) or 1
    named = [s for s in specs if s.record_type != "synthetic_negative"]
    m = len(named) or 1
    return {"cases": len(specs), "by_type": dict(collections.Counter(s.record_type for s in specs)),
            "mean_mentions_per_non_negative_note": {t: round(sum(s.counts[t] for s in named) / m, 2) for t in
                                                    ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]},
            "median_target_chars": sorted(s.target_chars for s in specs)[n // 2],
            "age_bands": dict(collections.Counter(s.age_band for s in specs)),
            "density": dict(collections.Counter(s.density_level for s in specs)),
            "gender": dict(collections.Counter(s.gender for s in specs))}
