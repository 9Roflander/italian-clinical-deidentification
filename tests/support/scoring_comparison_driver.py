"""Runs in a temporary tree holding one evaluator version. Invented notes only, with no access to real data.

Prints SHA-256 hashes of every generated result file, plus the numeric results of every constructed control.
"""
import hashlib
import json
import pathlib
import sys
import tempfile

from src.data.loaders import GoldNote
from src.evaluation.alignment import render_inline
from src.evaluation.controls import CONTROLS, baseline_e0_records, run_control
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.runner import GENERATED_FILES, build_outputs, evaluate_records, load_config, verify_run, write_run


def build_gold() -> VerifiedGold:
    notes, spans, fold_of = {}, {}, {}
    for k in range(20):
        name, other, age = f"Zelinda{chr(97 + k)}", "Quagliarino", str(21 + k)
        place = "Borgoverde dell'Aquilonia"
        date = f"{(k % 27) + 1}/03/20{10 + k}"
        if k == 7:
            text = "Nessun dato identificativo in questa nota, solo un decorso regolare."
            found = []
        else:
            text = (f"Città già nota: {name} {other}, di {age}  anni, vista il {date} a {place}; poi {name} è tornata (peso 18 kg, dose {age}0 mg) "
                    f"all'ambulatorio di {place}, più volte.")
            positions = [(text.index(f"{name} {other}"), len(f"{name} {other}"), "NOME"), (text.index(f"di {age} ") + 3, len(age), "ETÀ"),
                         (text.index(date), len(date), "DATA"), (text.index(place), len(place), "LUOGO/INDIRIZZO"),
                         (text.index(f"poi {name}") + 4, len(name), "NOME"), (text.rindex(place), len(place), "LUOGO/INDIRIZZO")]
            found = sorted((a, a + n, c) for a, n, c in positions)
        note_id = f"mini_{k:02d}"
        entities = tuple({"start": a, "end": b, "type": c, "text": text[a:b], "annotated_text": text[a:b]} for a, b, c in found)
        notes[note_id] = GoldNote(note_id, k, text, entities, redacted_text=render_inline(text, found))
        spans[note_id] = found
        fold_of[note_id] = k % 5 + 1
    return VerifiedGold(notes, tuple(notes), fold_of, spans, "mini-sha", "v2", {k: "x" for k in range(1, 6)})


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    gold = build_gold()
    cfg = load_config()
    cfg["bootstrap"]["iterations"] = 2000
    report = {"controls": {}, "control_files": {}}
    for name in CONTROLS:
        report["controls"][name] = run_control(name, gold)
        result = evaluate_records([CONTROLS[name](gold, i, name) for i in gold.order], gold)
        report["control_files"][name] = {k: sha(v) for k, v in build_outputs(result, gold, cfg).items()}
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run = write_run(baseline_e0_records(gold), gold, cfg, base / "public", base / "secure", stage="evaluator_control")
        report["e0_files"] = {name: sha((run / name).read_text(encoding="utf-8")) for name in GENERATED_FILES}
        verified = verify_run(run, gold, cfg, base / "secure")
        report["e0_reproduced"], report["e0_problems"] = verified["reproduced"], verified["problems"]
        report["e0_manifest_keys"] = sorted(json.loads((run / "run_manifest.json").read_text(encoding="utf-8")))
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
