"""Raw output preservation, parsing, truncation, and compatibility with evaluator v1.0.1, using a fake generator."""
import hashlib
import json

import pytest

import src.training.inference as inference
from src.data.loaders import GoldNote
from src.data.splits import FoldSpec, LeakageError
from src.evaluation.alignment import EMPTY, INVALID_JSON, SEGMENTS_MISMATCH, TRUNCATED, VALID, render_inline
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.metrics import as_report, compute_metrics
from src.evaluation.runner import evaluate_records, load_config, read_predictions, verify_run, write_run
from src.evaluation.scoring import totals
from src.training.dataset import Example
from src.training.formatting import EOT
from src.training.inference import InferenceError, check_limit_covers_targets, load_held_out_notes, max_new_tokens_for, merge_out_of_fold, run_fold_inference, stop_token_ids
from src.training.targets import build_segments, serialize_target
from tests.support.byte_tokenizer import ByteTokenizer
from tests.support.training_project import make_project

TOK = ByteTokenizer()
SYSTEM = "Istruzioni di prova."
SETTINGS = {"max_new_tokens_factor": 2.0, "max_new_tokens_margin": 512}
META = {"experiment": "E1", "model": "m", "model_revision": "r", "checkpoint": "E1/seed_42", "config_id": "c", "seed": 42}


def make_note(i: int):
    name, age = f"Zelinda{chr(97 + i)}", str(30 + i)
    text = f"{name} ha {age}  anni, città."
    spans = [(0, len(name), "NOME"), (len(name) + 4, len(name) + 6, "ETÀ")]
    return GoldNote(f"mini_{i:02d}", i, text, tuple({"start": a, "end": b, "type": c, "text": text[a:b], "annotated_text": text[a:b]} for a, b, c in spans), redacted_text=render_inline(text, spans)), spans


NOTES = [make_note(i) for i in range(10)]
BY_ID = {n.note_id: (n, s) for n, s in NOTES}


def spec_for(fold: int) -> FoldSpec:
    test = tuple(n.note_id for n, _ in NOTES if n.index % 5 + 1 == fold)
    train = tuple(n.note_id for n, _ in NOTES if n.note_id not in test)
    return FoldSpec(fold, train, test, "mini", 42)


def gold() -> VerifiedGold:
    return VerifiedGold({n.note_id: n for n, _ in NOTES}, tuple(n.note_id for n, _ in NOTES), {n.note_id: n.index % 5 + 1 for n, _ in NOTES}, {n.note_id: s for n, s in NOTES}, "mini", "v2", {k: "x" for k in range(1, 6)})


class Scripted:
    """Answers by finding which note is in the prompt."""

    def __init__(self, answer):
        self.answer, self.limits, self.prompts = answer, [], []

    def generate(self, prompt_ids, max_new_tokens):
        self.limits.append(max_new_tokens)
        prompt = TOK.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.prompts.append(prompt)
        note = next(n for n, _ in NOTES if n.text in prompt)
        return TOK(self.answer(note))["input_ids"][:max_new_tokens]


def perfect(note):
    return serialize_target(build_segments(note.text, BY_ID[note.note_id][1])) + EOT


def run(tmp_path, answer, fold=1, name="fold"):
    spec = spec_for(fold)
    notes = [BY_ID[i][0] for i in spec.test_ids]
    return run_fold_inference(Scripted(answer), TOK, SYSTEM, notes, spec, META, SETTINGS, tmp_path / f"{name}_{fold}")


def rows(folder, name="predictions.jsonl"):
    return [json.loads(x) for x in (folder / name).read_text(encoding="utf-8").splitlines()]


def scores_for(records):
    return evaluate_records(records, gold()).scores


def test_perfect_output_is_complete_and_parses(tmp_path):
    out = run(tmp_path, perfect)
    r = rows(out)
    assert all(x["parse_status"] == VALID and x["completion_status"] == "complete" and x["finish_reason"] == "stop" and x["truncation_flag"] is False for x in r)
    assert all(x["generation_parameters"]["do_sample"] is False and x["generation_parameters"]["whitespace_cleanup"] is False for x in r)


def test_raw_output_is_saved_before_any_parsing(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("parser failed")
    monkeypatch.setattr(inference, "resolve_segments", broken)
    with pytest.raises(RuntimeError):
        run(tmp_path, perfect)
    raw = rows(tmp_path / "fold_1", "raw_predictions.jsonl")
    assert len(raw) == 1 and raw[0]["raw_model_output"] and raw[0]["generated_token_ids"]
    assert not (tmp_path / "fold_1" / "predictions.jsonl").exists()


@pytest.mark.parametrize("text", ["  testo con spazi esterni  \n", 'prima\n\n```json\n[{"text":"x","label":"O"}]\n```', "   ", "Testo  con  doppi  spazi"])
def test_no_cleanup_is_ever_applied_to_raw_output(tmp_path, text):
    out = run(tmp_path, lambda note: text + EOT)
    assert all(x["raw_model_output"] == text for x in rows(out)) and all(x["raw_model_output"] == text for x in rows(out, "raw_predictions.jsonl"))


def test_raw_file_keeps_the_generated_tokens(tmp_path):
    out = run(tmp_path, perfect)
    for r in rows(out, "raw_predictions.jsonl"):
        assert r["generated_token_ids"][-1] in stop_token_ids(TOK)
        assert TOK.decode(r["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) == r["raw_model_output"]


def test_hitting_the_token_limit_is_marked_truncated_and_earns_no_credit(tmp_path):
    long_answer = lambda note: "x" * 5000
    out = run(tmp_path, long_answer)
    r = rows(out)
    assert all(x["completion_status"] == "truncated" and x["finish_reason"] == "length" and x["truncation_flag"] is True for x in r)
    assert all(s.status == TRUNCATED for s in scores_for(r) if s.note_id in {x["note_id"] for x in r})


def test_even_a_complete_looking_output_at_the_limit_is_truncated(tmp_path):
    out = run(tmp_path, lambda note: serialize_target(build_segments(note.text, BY_ID[note.note_id][1])))
    r = rows(out)
    assert all(x["truncation_flag"] is True for x in r), "no stop token means no proof the model finished"
    assert all(s.status == TRUNCATED for s in scores_for(r) if s.note_id in {x["note_id"] for x in r})


@pytest.mark.parametrize("answer,status", [
    ("", EMPTY),
    ('[{"text":"Zelinda","label":"NOME"},', INVALID_JSON),
    ("not json at all", INVALID_JSON),
    ('[{"text":"parole diverse","label":"O"}]', SEGMENTS_MISMATCH),
    ('{"a":1}', INVALID_JSON),
])
def test_invalid_outputs_are_detected_and_scored_with_no_credit(tmp_path, answer, status):
    out = run(tmp_path, lambda note: answer + EOT)
    r = rows(out)
    assert all(x["parse_status"] == status for x in r)
    fold_ids = {x["note_id"] for x in r}
    for s in scores_for(r):
        if s.note_id in fold_ids:
            assert s.status == status and s.counts["tp_NOME"] == 0 and s.counts["covered_chars"] == 0


def test_whitespace_change_in_the_copy_is_invalid(tmp_path):
    def collapsed(note):
        return serialize_target([{"text": note.text.replace("  ", " "), "label": "O"}]) + EOT
    out = run(tmp_path, collapsed)
    assert all(x["parse_status"] != VALID for x in rows(out))


def test_token_limit_scales_with_the_note():
    assert max_new_tokens_for(100, 2.0, 64) == 264 and max_new_tokens_for(101, 1.5, 0) == 152


def many_entity_example(n_entities: int) -> Example:
    """A short note where JSON overhead outweighs the text, which a proportional limit would cut off."""
    words = [f"z{i}" for i in range(n_entities)]
    text = " ".join(words)
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w), "NOME"))
        pos += len(w) + 1
    return Example("dense", "gold", "dense", 0, "gold", text, serialize_target(build_segments(text, spans)))


def test_a_limit_too_small_for_dense_notes_is_refused_before_training():
    dense = many_entity_example(15)
    with pytest.raises(InferenceError, match="exceed the inference token limit"):
        check_limit_covers_targets(TOK, [dense], {"max_new_tokens_factor": 2.0, "max_new_tokens_margin": 64})


def test_the_configured_limit_covers_a_dense_note_and_reports_headroom():
    """Bytes overstate JSON overhead compared with real tokens. Real headroom is measured when training starts."""
    from src.training.experiments import TRAINING_CONFIG, load_yaml
    report = check_limit_covers_targets(TOK, [many_entity_example(8)], load_yaml(TRAINING_CONFIG)["inference"])
    assert report["smallest_headroom_tokens"] > 0 and report["tightest_example_id"] == "dense"


def test_generator_receives_the_computed_limit_and_only_the_note(tmp_path):
    spec = spec_for(1)
    notes = [BY_ID[i][0] for i in spec.test_ids]
    g = Scripted(perfect)
    run_fold_inference(g, TOK, SYSTEM, notes, spec, META, SETTINGS, tmp_path / "x")
    assert g.limits == [max_new_tokens_for(len(TOK(n.text)["input_ids"]), 2.0, 512) for n in sorted(notes, key=lambda n: n.note_id)]
    assert all('"label":"NOME"' not in p for p in g.prompts)


def test_notes_outside_the_held_out_set_are_refused(tmp_path):
    spec = spec_for(1)
    wrong = [BY_ID[i][0] for i in spec_for(2).test_ids]
    with pytest.raises(LeakageError):
        run_fold_inference(Scripted(perfect), TOK, SYSTEM, wrong, spec, META, SETTINGS, tmp_path / "x")


def test_partial_held_out_set_is_refused(tmp_path):
    spec = spec_for(1)
    with pytest.raises(LeakageError):
        run_fold_inference(Scripted(perfect), TOK, SYSTEM, [BY_ID[spec.test_ids[0]][0]], spec, META, SETTINGS, tmp_path / "x")


def test_prediction_folders_cannot_be_overwritten(tmp_path):
    run(tmp_path, perfect)
    with pytest.raises(InferenceError, match="immutable"):
        run(tmp_path, perfect)


def test_missing_metadata_is_refused(tmp_path):
    spec = spec_for(1)
    with pytest.raises(InferenceError, match="metadata"):
        run_fold_inference(Scripted(perfect), TOK, SYSTEM, [BY_ID[i][0] for i in spec.test_ids], spec, {"model": "m"}, SETTINGS, tmp_path / "x")


def test_prediction_manifest_is_complete_and_holds_no_text(tmp_path):
    out = run(tmp_path, perfect)
    manifest = json.loads((out / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"] == 2 and manifest["fold"] == 1 and manifest["parse_status_counts"] == {VALID: 2} and manifest["finish_reason_counts"] == {"stop": 2}
    assert len(manifest["raw_predictions_sha256"]) == len(manifest["predictions_sha256"]) == 64 and manifest["settings"] == SETTINGS
    assert "Zelinda" not in json.dumps(manifest, ensure_ascii=False)


def test_reruns_with_the_same_generator_give_identical_files(tmp_path):
    a, b = run(tmp_path, perfect, name="a"), run(tmp_path, perfect, name="b")
    for name in ("predictions.jsonl", "raw_predictions.jsonl"):
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == hashlib.sha256((b / name).read_bytes()).hexdigest()


def all_folds(tmp_path, answer=perfect):
    return [run(tmp_path, answer, k) for k in range(1, 6)]


def test_merged_predictions_score_perfectly_in_the_evaluator(tmp_path):
    summary = merge_out_of_fold(all_folds(tmp_path), tmp_path / "merged.jsonl")
    assert summary["records"] == 10 and summary["folds"] == [1, 2, 3, 4, 5]
    result = evaluate_records(read_predictions(tmp_path / "merged.jsonl"), gold())
    metrics = as_report(compute_metrics(totals(result.scores)))
    assert metrics["micro_f1"] == 1 and metrics["macro_f1"] == 1 and metrics["invalid_output_rate"] == 0 and metrics["note_exact_rate"] == 1


def test_a_full_evaluator_run_reproduces_from_merged_predictions(tmp_path):
    merge_out_of_fold(all_folds(tmp_path), tmp_path / "merged.jsonl")
    cfg = load_config()
    cfg["bootstrap"]["iterations"] = 2000
    run_dir = write_run(read_predictions(tmp_path / "merged.jsonl"), gold(), cfg, tmp_path / "experiments/development/e", tmp_path / "secure")
    report = verify_run(run_dir, gold(), cfg, tmp_path / "secure")
    assert report["reproduced"] and json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["official_run"] is False


def test_merge_refuses_a_changed_file(tmp_path):
    folds = all_folds(tmp_path)
    with open(folds[0] / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(InferenceError, match="differ"):
        merge_out_of_fold(folds, tmp_path / "m.jsonl")


def test_merge_refuses_a_repeated_fold(tmp_path):
    folds = all_folds(tmp_path)
    with pytest.raises(InferenceError, match="twice"):
        merge_out_of_fold([folds[0], folds[0]], tmp_path / "m.jsonl")


def test_merge_refuses_mixed_runs(tmp_path):
    a = run(tmp_path, perfect, 1)
    spec = spec_for(2)
    b = run_fold_inference(Scripted(perfect), TOK, SYSTEM, [BY_ID[i][0] for i in spec.test_ids], spec, {**META, "seed": 7}, SETTINGS, tmp_path / "other")
    with pytest.raises(InferenceError, match="one experiment"):
        merge_out_of_fold([a, b], tmp_path / "m.jsonl")


def test_merge_refuses_to_overwrite(tmp_path):
    folds = all_folds(tmp_path)
    merge_out_of_fold(folds, tmp_path / "m.jsonl")
    with pytest.raises(InferenceError, match="exists"):
        merge_out_of_fold(folds, tmp_path / "m.jsonl")


def test_the_evaluator_refuses_a_prediction_from_the_wrong_fold_model(tmp_path):
    out = run(tmp_path, perfect)
    r = rows(out)
    r[0]["fold_id"] = 3
    with pytest.raises(LeakageError):
        evaluate_records(r, gold())


def test_held_out_loader_returns_only_the_fold_and_checks_the_gold_hash(tmp_path):
    project = make_project(tmp_path)
    spec = project["specs"][2]
    assert {n.note_id for n in load_held_out_notes(spec, project["gold"])} == set(spec.test_ids)
    project["gold"].write_text(project["gold"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(LeakageError):
        load_held_out_notes(spec, project["gold"])
