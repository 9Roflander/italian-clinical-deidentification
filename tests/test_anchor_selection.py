"""Guide test 2: no test note reaches a prompt, as an anchor or as a hard-coded example."""
import inspect
import re

import pytest

import src.generation.anchor_selection as anchor_mod
from src.data.annotations import canonical_type, word_boundary_regex
from src.data.loaders import load_gold
from src.data.splits import LeakageError, freeze_folds, load_fold, load_train_notes
from src.generation.anchor_selection import AnchorSelector, eligible_pool
from src.generation.generation_plan import build_plan, plan_sha
from src.generation.prompt_templates import PROMPTS_DIR, read_template
from src.generation.synthetic_generator import build_request_prompt, group_requests
from tests.conftest import make_gold, sentinel


def selector_for(project, k):
    s = project["specs"][k - 1]
    return s, load_train_notes(s, project["gold"]), AnchorSelector(s, load_train_notes(s, project["gold"]))


@pytest.mark.parametrize("k", range(1, 6))
def test_real_anchors_come_only_from_the_training_partition(k):
    s = load_fold(k)
    sel = AnchorSelector(s, load_train_notes(s))
    used = {a.note_id for i in range(3000) for a in sel.for_request(i)}
    assert used <= set(s.train_ids)
    assert not used & set(s.test_ids)
    assert used == set(sel.pool_ids)


def test_each_request_gets_one_short_one_medium_one_long_note():
    s = load_fold(1)
    sel = AnchorSelector(s, load_train_notes(s))
    for i in range(50):
        a = sel.for_request(i)
        assert len(a) == 3 and len(a[0].text) <= len(a[1].text) <= len(a[2].text)


def test_anchors_actually_rotate():
    s = load_fold(1)
    sel = AnchorSelector(s, load_train_notes(s))
    assert len({tuple(a.note_id for a in sel.for_request(i)) for i in range(60)}) > 5


def test_test_note_offered_as_material_is_refused(project):
    s = project["specs"][0]
    leaked = [n for n in load_gold(project["gold"]) if n.note_id == s.test_ids[0]]
    with pytest.raises(LeakageError):
        AnchorSelector(s, load_train_notes(s, project["gold"]) + leaked)


@pytest.mark.parametrize("k", range(1, 6))
def test_no_test_sentinel_in_any_rendered_prompt(project, k):
    s, train, sel = selector_for(project, k)
    test_tokens = [sentinel(int(i.split("_")[1])) for i in s.test_ids]
    plan = build_plan(k, train, project["cfg"])
    seen_train_sentinel = False
    for g in group_requests(plan, 1, 3):
        for attempt_index in range(40):
            g2 = dict(g, index=g["index"] + attempt_index)
            prompt, _ = build_request_prompt(g2, "style_anchored", sel, project["cfg"], project["prompts"])
            ok = not any(t in prompt for t in test_tokens)
            assert ok, f"fold {k}: test sentinel in prompt"
            seen_train_sentinel = seen_train_sentinel or "ZZSENTINEL" in prompt
    assert seen_train_sentinel, "test is vacuous: anchors never reached the prompt"


def test_plan_does_not_change_when_test_notes_change(project, tmp_path):
    import json
    s = project["specs"][0]
    gold = make_gold()
    first = plan_sha(build_plan(1, load_train_notes(s, project["gold"]), project["cfg"]))
    for tid in s.test_ids:
        i = int(tid.split("_")[1])
        gold[i] = {"text": "testo del tutto diverso " * 40, "entities": [{"type": "NOME", "text": "diverso"}] * 9}
    other = tmp_path / "other.json"
    other.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    specs = freeze_folds(other, tmp_path / "s2", tmp_path / "f.csv")
    assert plan_sha(build_plan(1, load_train_notes(specs[0], other), project["cfg"])) == first


def test_templates_contain_no_concrete_values():
    for text in read_template("v1"):
        assert not re.search(r"\d", text), "digit in template"


def test_templates_contain_no_gold_entity_string():
    static = "\n".join(read_template("v1"))
    hits = 0
    for n in load_gold():
        for e in n.entities:
            x = e.get("text", "")
            if canonical_type(e.get("type")) in ("NOME", "LUOGO/INDIRIZZO") and len(x) >= 3 and word_boundary_regex(x).search(static):
                hits += 1
    assert hits == 0


def test_config_spec_values_contain_no_gold_names():
    import yaml
    from src.data.loaders import ROOT
    cfg = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text(encoding="utf-8"))
    names = {e.get("text", "") for n in load_gold() for e in n.entities if canonical_type(e.get("type")) == "NOME"}
    used = set(cfg["plan"]["specialties"]) | set(cfg["plan"]["name_forms"])
    assert not names & used


def test_anchor_module_cannot_reach_the_full_gold_set():
    source = inspect.getsource(anchor_mod)
    assert "load_gold" not in source and "GOLD_PATH" not in source


def test_notes_with_unlocatable_annotations_are_not_anchors():
    from src.data.loaders import GoldNote
    bad = GoldNote("gold_000", 0, "testo senza il valore annotato", ({"type": "NOME", "text": "assente"},))
    good = GoldNote("gold_001", 1, "testo con presente incluso", ({"type": "NOME", "text": "presente"},))
    assert [n.note_id for n in eligible_pool([bad, good])] == ["gold_001"]
