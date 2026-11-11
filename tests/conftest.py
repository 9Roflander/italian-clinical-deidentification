"""Fixtures use invented text only. Real-data tests assert on booleans and IDs, so a
failure cannot print note text."""
import json
import pathlib
import random
import re

import pytest

from src.data import splits as splits_mod
from src.generation.gemini_client import GenerationResult

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILLER = ("il decorso clinico risulta regolare con progressivo miglioramento del quadro generale e buona risposta alla terapia "
          "impostata durante il ricovero presso la struttura di riferimento . ")


def sentinel(i: int) -> str:
    return f"ZZSENTINEL{i:03d}"


def make_gold(n=25):
    rng = random.Random(7)
    notes = []
    for i in range(n):
        name, place = f"Zelindo{chr(97 + i % 26)}{chr(97 + (i * 3) % 26)}", f"Paesello{chr(97 + i % 26)}{chr(98 + i % 24)}"
        age = str(30 + i)
        body = FILLER * rng.randint(3, 22)
        extra = {"NOME": [], "LUOGO/INDIRIZZO": []}
        text = f"Paziente di {age} anni osservato con codice {sentinel(i)} . "
        entities = [{"type": "ETÀ", "text": age}]
        if i % 2 == 0:
            text += f"{name} riferisce dolore toracico . "
            entities.append({"type": "NOME", "text": name})
        else:
            text += "Il paziente riferisce dolore toracico . "
        text += f"Viene ricoverato a {place} . "
        entities.append({"type": "LUOGO/INDIRIZZO", "text": place})
        notes.append({"text": text + body, "entities": entities})
    return notes


@pytest.fixture()
def project(tmp_path):
    """A complete miniature project: gold, frozen folds, real prompts, small config."""
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(make_gold(), ensure_ascii=False), encoding="utf-8")
    splits_dir = tmp_path / "splits"
    specs = splits_mod.freeze_folds(gold_path, splits_dir, tmp_path / "fold_assignments.csv")
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text(encoding="utf-8"))
    cfg["plan"].update({"n_named": 8, "n_name_free": 4, "n_negative": 3})
    cfg["limits"].update({"workers": 2, "requests_per_minute": 0, "max_cost_usd": 100})
    return {"gold": gold_path, "splits": splits_dir, "specs": specs, "cfg": cfg, "out": tmp_path / "synthetic",
            "prompts": ROOT / "prompts", "tmp": tmp_path}


def fake_case(case_id, counts, chars):
    """A valid case matching the requested counts, built only from what the prompt says."""
    tag = "".join(chr(97 + int(c)) if c.isdigit() else c for c in re.sub(r"\W", "", case_id).lower())
    segs = [{"text": "Caso clinico " + tag + " . ", "entity_type": "NESSUNA"}]
    for k in range(counts["NOME"]):
        segs += [{"text": f"Nominativo{tag.capitalize()}", "entity_type": "NOME"}, {"text": " riferisce dolore . ", "entity_type": "NESSUNA"}]
    for k in range(counts["ETÀ"]):
        segs += [{"text": "Soggetto con età pari a ", "entity_type": "NESSUNA"}, {"text": str(21 + k), "entity_type": "ETÀ"}, {"text": " . ", "entity_type": "NESSUNA"}]
    for k in range(counts["DATA"]):
        segs += [{"text": "Primo accesso nel ", "entity_type": "NESSUNA"}, {"text": f"marzo {2001 + k}", "entity_type": "DATA"}, {"text": " . ", "entity_type": "NESSUNA"}]
    for k in range(counts["LUOGO/INDIRIZZO"]):
        segs += [{"text": "Trasferito presso ", "entity_type": "NESSUNA"}, {"text": f"Borgo{tag.capitalize()}x{chr(97 + k)}", "entity_type": "LUOGO/INDIRIZZO"}, {"text": " . ", "entity_type": "NESSUNA"}]
    used = sum(len(s["text"]) for s in segs)
    filler = f"osservazione {tag} con valutazione specialistica e controllo periodico senza eventi avversi di rilievo . "
    segs.append({"text": filler * max(1, (chars - used) // len(filler)), "entity_type": "NESSUNA"})
    return {"case_id": case_id, "segments": segs}


class FakeClient:
    """Stands in for Gemini, reading its instructions from the rendered prompt."""
    model_id = "fake"
    sdk_version = "fake"

    def __init__(self, broken_ids=(), raise_after=None, raise_with=None):
        self.prompts, self.seeds, self.broken, self.raise_after, self.raise_with = [], [], set(broken_ids), raise_after, raise_with

    def generate(self, prompt, seed):
        if self.raise_after is not None and len(self.prompts) >= self.raise_after:
            raise self.raise_with
        self.prompts.append(prompt)
        self.seeds.append(seed)
        cases = []
        blocks = re.findall(r"SCHEDA (F\d-N\d+)\n(.*?)(?=\n\nSCHEDA |\Z)", prompt, flags=re.S)
        for cid, block in blocks:
            chars = int(re.search(r"circa (\d+) caratteri", block).group(1))
            m = re.search(r"NOME (\d+), ETÀ (\d+), DATA (\d+), LUOGO/INDIRIZZO (\d+)", block)
            counts = dict(zip(["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"], map(int, m.groups())))
            if cid in self.broken:
                self.broken.discard(cid)
                case = fake_case(cid, counts, chars)
                case["segments"].append({"text": " Contatto del 12/03/2015 .", "entity_type": "NESSUNA"})
                cases.append(case)
            else:
                cases.append(fake_case(cid, counts, chars))
        return GenerationResult(json.dumps({"cases": cases}, ensure_ascii=False), "STOP", "fake-001", "resp", 100, 200, 10, 310, 0.01)
