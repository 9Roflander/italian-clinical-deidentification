"""Generation v2: regression tests for the seven approved changes, and proof that v1 is unchanged. Invented data only."""
import collections
import copy
import json
import random
import re
from dataclasses import asdict

import pytest
import yaml

from src.data.loaders import GOLD_V2, ROOT
from src.generation import synthetic_generator as g
from src.generation.generation_plan import V2_FIELDS, band_limits, build_plan, plan_sha, year_range
from src.generation.medical_eponyms import capitalised_word_counts
from src.generation.post_audit import audit_fold
from src.generation.prompt_templates import read_template, render_case_spec, render_prompt, template_sha
from src.generation.synthetic_validation import ValidationContext, build_context, validate_case
from src.generation.gemini_client import GenerationResult

V1_SHA = "b244070a14accb6858c157689458acef1b999e570b5aa58dff13b1b799b07334"
V2_SHA = "c612f46ba8d8e54d287f991099f839544c54e8dd3dd2477b32936af179e842ee"
V2 = yaml.safe_load((ROOT / "configs" / "generation_v2.yaml").read_text(encoding="utf-8"))
V1 = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text(encoding="utf-8"))
VOCAB = ("riscontro esame valutazione decorso terapia controllo paziente quadro sintomi dolore febbre tosse ricovero reparto indagine "
         "ecografia risonanza biopsia diagnosi trattamento miglioramento stabile lieve moderato marcato progressivo acuto cronico addome torace "
         "cuore polmone rene fegato cute arto ginocchio spalla testa collo respiro pressione frequenza saturazione emoglobina creatinina sodio "
         "potassio glicemia leucociti piastrine farmaco dose somministrazione giorno settimana visita ambulatorio sintomatologia episodio "
         "insorgenza durata intensità localizzazione irradiazione palpazione auscultazione percussione ispezione obiettivo strumentale laboratorio "
         "referto immagine lesione nodulo versamento edema infiammazione infezione antibiotico steroide analgesico idratazione riposo dieta").split()


def v2_cfg(project_cfg):
    cfg = copy.deepcopy(V2)
    cfg["plan"].update({"n_named": 8, "n_name_free": 4, "n_negative": 3})
    cfg["limits"].update({"workers": 2, "requests_per_minute": 0, "max_cost_usd": 100})
    return cfg


def words(rng, n):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def diverse_case(case_id, block, copy_from=None):
    """A varied invented note that follows its specification, with digits for measurements and no line breaks."""
    rng = random.Random(case_id)
    chars = int(re.search(r"circa (\d+) caratteri", block).group(1))
    counts = dict(zip(["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"], map(int, re.search(r"NOME (\d+), ETÀ (\d+), DATA (\d+), LUOGO/INDIRIZZO (\d+)", block).groups())))
    age = re.search(r"Età del paziente: (\d+) anni", block)
    tag = "".join(chr(97 + int(c)) for c in re.sub(r"\D", "", case_id))
    segs = [{"text": f"Caso {tag} con {words(rng, 6)}, pressione {rng.randint(100, 160)}/{rng.randint(60, 95)} mmHg, {words(rng, 3)}, emoglobina {rng.randint(100, 160) / 10} g/dL. ", "entity_type": "NESSUNA"}]
    ages = [age.group(1)] if age else []
    ages += [str(21 + k) for k in range(counts["ETÀ"] - len(ages))]
    for k in range(counts["NOME"]):
        segs += [{"text": f"Nominato{tag}{chr(97 + k)}", "entity_type": "NOME"}, {"text": f" riferisce {words(rng, 5)}. ", "entity_type": "NESSUNA"}]
    for a in ages:
        segs += [{"text": f"{words(rng, 3)} di ", "entity_type": "NESSUNA"}, {"text": a, "entity_type": "ETÀ"}, {"text": f" anni, {words(rng, 4)}. ", "entity_type": "NESSUNA"}]
    for k in range(counts["DATA"]):
        segs += [{"text": f"Accesso {words(rng, 2)} nel ", "entity_type": "NESSUNA"}, {"text": f"marzo {2001 + k}", "entity_type": "DATA"}, {"text": ". ", "entity_type": "NESSUNA"}]
    for k in range(counts["LUOGO/INDIRIZZO"]):
        segs += [{"text": "Trasferito presso ", "entity_type": "NESSUNA"}, {"text": f"Borgo{tag}x{chr(97 + k)}", "entity_type": "LUOGO/INDIRIZZO"}, {"text": f", {words(rng, 3)}. ", "entity_type": "NESSUNA"}]
    while sum(len(s["text"]) for s in segs) < chars:
        segs.append({"text": f"{words(rng, 8)} con valore di {rng.randint(101, 999)} mg. ", "entity_type": "NESSUNA"})
    if copy_from is not None:
        segs = copy_from
    return {"case_id": case_id, "segments": segs}


class DiverseClient:
    model_id, sdk_version = "fake", "fake"

    def __init__(self, mutate=None):
        self.prompts, self.mutate, self.first = [], mutate, None

    def generate(self, prompt, seed):
        self.prompts.append(prompt)
        cases = []
        for cid, block in re.findall(r"SCHEDA (F\d-N\d+)\n(.*?)(?=\n\nSCHEDA |\Z)", prompt, flags=re.S):
            case = diverse_case(cid, block)
            if self.mutate:
                case = self.mutate(self, cid, case)
            cases.append(case)
        return GenerationResult(json.dumps({"cases": cases}, ensure_ascii=False), "STOP", "fake-001", "resp", 100, 200, 0, 300, 0.01)


def run_v2(project, client, **kw):
    return g.run_generation(v2_cfg(project["cfg"]), 1, "non_anchored", execute=True, output_root=project["out"], client=client,
                            gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"], **kw)


def log(project):
    p = project["out"] / "fold_1" / "non_anchored" / "case_log.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


# Versions and hashes

def test_v1_prompt_and_hash_are_preserved():
    assert template_sha("v1") == V1_SHA and V1["prompt_version"] == "v1" and V1["generation_version"] == "v1"


def test_v2_prompt_hash_is_recorded_and_config_points_to_it():
    assert template_sha("v2") == V2_SHA and V2["prompt_version"] == "v2" and V2["generation_version"] == "v2"


def test_v2_template_has_no_concrete_values():
    for text in read_template("v2"):
        assert not re.search(r"\d", text)


def test_the_cli_and_audit_default_to_v2():
    assert "generation_v2.yaml" in (ROOT / "src/generation/synthetic_generator.py").read_text(encoding="utf-8")
    assert "generation_v2.yaml" in (ROOT / "src/generation/post_audit.py").read_text(encoding="utf-8")


def test_unchanged_settings():
    for key in ("cases_per_request",):
        assert V2["plan"][key] == V1["plan"][key] == 3
    assert V2["model"] == V1["model"] and V2["model"]["id"] == "gemini-3.8-flash" and V2["model"]["temperature"] == 0.7
    assert V2["anchors"] == V1["anchors"]
    for key in ("count_tolerance_share", "count_tolerance_min", "max_verbatim_run_anchor", "max_verbatim_run_train", "near_duplicate_containment", "min_entities"):
        assert V2["validation"][key] == V1["validation"][key], key


def test_reduced_plan_counts():
    assert (V2["plan"]["n_named"], V2["plan"]["n_name_free"], V2["plan"]["n_negative"]) == (202, 78, 20)


# 1 Digits for measurements, 2 one paragraph, 3 actual ages, 5 diversity: prompt instructions

def test_prompt_v2_instructions():
    main = read_template("v2")[0]
    assert "in cifre, con le unità di misura" in main and "anche per i casi senza informazioni sensibili" in main
    assert "Un unico paragrafo continuo" in main
    assert "l'età reale di una persona" in main and "età future o pianificate" in main
    assert "Evita formule d'apertura ricorrenti" in main and "diverse tra loro" in main
    old = read_template("v1")[0]
    assert "Un unico paragrafo continuo" not in old


# 4 Reference years and exact ages

class Note:
    def __init__(self, note_id, entities):
        self.note_id, self.entities, self.text = note_id, entities, "x" * 900


def test_year_range_comes_from_training_dates_with_a_fallback():
    details = V2["plan"]["details"]
    notes = [Note(f"n{i}", [{"type": "DATA", "text": f"marzo {y}"}]) for i, y in enumerate([2003, 2007, 2011, 2015, 2019])]
    assert year_range(notes, details) == (2003, 2019, "training_partition_dates")
    assert year_range(notes[:2], details) == (1995, 2024, "configured_fallback")


def test_age_limits_follow_the_band_label():
    assert band_limits({"label": "66-95", "min": 66, "max": 120}) == (66, 95) and band_limits({"label": "0-17", "min": 0, "max": 17}) == (1, 17)


def test_v2_plan_samples_exact_ages_and_years_only_where_needed(project):
    from src.data.splits import load_fold, load_train_notes
    train = load_train_notes(load_fold(1, project["splits"]), project["gold"])
    specs = build_plan(1, train, v2_cfg(project["cfg"]))
    for s in specs:
        if s.record_type == "synthetic_negative" or not s.n_eta:
            assert s.patient_age is None
        else:
            low, high = band_limits({"label": s.age_band})
            assert low <= s.patient_age <= high
        assert (s.reference_year is not None) == bool(s.n_data)
    assert build_plan(1, train, v2_cfg(project["cfg"])) == specs


def test_v2_details_do_not_change_the_other_plan_fields(project):
    from src.data.splits import load_fold, load_train_notes
    train = load_train_notes(load_fold(1, project["splits"]), project["gold"])
    with_details, without = v2_cfg(project["cfg"]), v2_cfg(project["cfg"])
    del without["plan"]["details"]
    a, b = build_plan(1, train, with_details), build_plan(1, train, without)
    strip = lambda s: {k: v for k, v in asdict(s).items() if k not in V2_FIELDS}
    assert [strip(x) for x in a] == [strip(x) for x in b]


def test_v1_plans_specs_and_hashes_are_unchanged(project):
    from src.data.splits import load_fold, load_train_notes
    train = load_train_notes(load_fold(1, project["splits"]), project["gold"])
    specs = build_plan(1, train, project["cfg"])
    assert all(s.reference_year is None and s.patient_age is None for s in specs)
    legacy = __import__("hashlib").sha256(json.dumps([{k: v for k, v in asdict(s).items() if k not in V2_FIELDS} for s in specs], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert plan_sha(specs) == legacy
    assert all("Età del paziente" not in render_case_spec(s) and "Anno di riferimento" not in render_case_spec(s) for s in specs)


@pytest.mark.skipif(not GOLD_V2.exists(), reason="needs the secure gold file")
def test_real_v1_pilot_plan_hash_is_unchanged():
    from src.data.splits import load_fold, load_train_notes
    recorded = json.loads((ROOT / "experiments/development/non_anchored_pilot/fold_1/non_anchored/run_config.json").read_text(encoding="utf-8"))["plan_sha256"]
    assert plan_sha(build_plan(1, load_train_notes(load_fold(1)), V1)) == recorded


def test_v2_spec_lines():
    from src.generation.generation_plan import CaseSpec
    s = CaseSpec("F1-N00001", 1, "synthetic_named", 1, 0, "cardiologia", "donna", "41-65", "Lazio", "nome proprio", "A", 1, 1, 1, 0, 900, "bassa", 2009, 52)
    text = render_case_spec(s)
    assert "- Età del paziente: 52 anni" in text and "- Anno di riferimento: 2009" in text
    neg = CaseSpec("F1-N00002", 1, "synthetic_negative", 2, 0, "cardiologia", "donna", "41-65", "Lazio", "nome proprio", "A", 0, 0, 0, 0, 900, "nessuna")
    assert "Età del paziente" not in render_case_spec(neg) and "Anno di riferimento" not in render_case_spec(neg)


# Validator rules on single cases

def context(cfg, accepted=None):
    c = build_context([], cfg)
    ctx = ValidationContext(c["train_hashes"], c["train_token_docs"], c["train_index"], [], set(), cfg)
    for k, text in (accepted or {}).items():
        g.remember_accepted({"accepted_tokens": ctx.accepted_tokens, "accepted_index": ctx.accepted_index}, k, text)
    return ctx


def spec_for(n_eta=2, age=52, n_nome=0, chars=400, kind="synthetic_name_free"):
    from src.generation.generation_plan import CaseSpec
    return CaseSpec("F1-N00001", 1, kind, 1, 0, "cardiologia", "donna", "41-65", "Lazio", "nome proprio", "A", n_nome, n_eta, 0, 0, chars, "bassa", None, age)


def case_from(text_parts):
    return {"case_id": "F1-N00001", "segments": [{"text": t, "entity_type": k} for t, k in text_parts]}


FILLER = ("Esame obiettivo nella norma, saturazione 97%, frequenza 78 battiti al minuto, terapia con 500 mg ogni 8 ore e miglioramento progressivo dei sintomi riferiti. "
          "Controllo ecografico regolare, creatinina 0,9 mg/dL, sodio 139 mmol/L, potassio 4,1 mmol/L e decorso clinico favorevole durante la degenza in reparto. ")
BASE = [("Paziente di ", "NESSUNA"), ("52", "ETÀ"), (" anni con pressione 130/80 mmHg ed emoglobina 12,4 g/dL; la sorella di ", "NESSUNA"), ("49", "ETÀ"), (" anni sta bene. " + FILLER * 2, "NESSUNA")]


def reasons(parts, cfg=None, spec=None, accepted=None):
    cfg = cfg or V2
    return validate_case(case_from(parts), spec or spec_for(chars=len("".join(t for t, _ in parts))), context(cfg, accepted)).reasons


def test_base_case_is_valid_under_v2():
    assert reasons(BASE) == []


@pytest.mark.parametrize("brk", ["\n", "\r\n", " ", "\n\n"])
def test_2_line_breaks_are_rejected_in_v2(brk):
    parts = BASE[:4] + [(BASE[4][0][:40] + brk + BASE[4][0][40:], "NESSUNA")]
    assert "line_break_in_text" in reasons(parts)
    assert "line_break_in_text" not in reasons(parts, V1), "v1 behaviour is unchanged"


@pytest.mark.parametrize("prefix", ["fino al compimento dei ", "al raggiungimento dei ", "entro i ", "superati i ", "fino ai "])
def test_3_future_milestone_ages_are_rejected(prefix):
    parts = [("Paziente di ", "NESSUNA"), ("52", "ETÀ"), (f" anni, controllo programmato {prefix}", "NESSUNA"), ("60", "ETÀ")] + BASE[2:]
    assert "age_milestone_marked" in reasons(parts, spec=spec_for(n_eta=3, chars=len("".join(t for t, _ in parts))))


@pytest.mark.parametrize("prefix", ["Paziente di ", "all'età di ", "una sorella di ", "di anni "])
def test_3_actual_ages_are_accepted(prefix):
    parts = [(prefix, "NESSUNA"), ("52", "ETÀ")] + BASE[2:]
    assert "age_milestone_marked" not in reasons(parts) and reasons(parts) == []


def test_4_the_specified_patient_age_must_be_marked():
    parts = [("Paziente di ", "NESSUNA"), ("47", "ETÀ")] + BASE[2:]
    assert "patient_age_missing" in reasons(parts)
    assert "patient_age_missing" not in reasons(BASE)


def test_6_near_duplicate_of_an_accepted_note_is_rejected():
    accepted = {"F1-N00099": "".join(t for t, _ in BASE)}
    assert "near_duplicate_of_accepted_note" in reasons(BASE, accepted=accepted)
    different = {"F1-N00099": "testo del tutto diverso " * 30}
    assert "near_duplicate_of_accepted_note" not in reasons(BASE, accepted=different)
    assert "near_duplicate_of_accepted_note" not in reasons(BASE, V1, accepted=accepted), "v1 has no such rule"


def test_6_a_run_of_eleven_words_is_allowed_and_twelve_is_not():
    shared = "uno due tre quattro cinque sei sette otto nove dieci undici dodici".split()
    note = lambda k, tail: " ".join(shared[:k]) + " " + tail
    a = note(12, "alfa beta gamma delta " * 20)
    for k, expected in ((11, False), (12, True)):
        parts = [("Paziente di ", "NESSUNA"), ("52", "ETÀ"), (" anni. " + note(k, "zeta eta theta iota " * 20), "NESSUNA")]
        got = "near_duplicate_of_accepted_note" in reasons(parts, accepted={"F1-N00099": a})
        assert got is expected, k


# 7 Capitalized-word flag

@pytest.mark.parametrize("text,unexplained", [
    ("diagnosi di morbo di Basedow confermata", 0),
    ("positivo alle manovre di Neer, Hawkins-Kennedy e Jobe in sede", 0),
    ("calcolato con la formula di Bazett e poi", 0),
    ("tiroidite cronica autoimmune di Hashimoto in fase", 0),
    ("monitoraggio Holter delle ventiquattro ore", 0),
    ("ricoverato il signor Parkinson per dolore", 1),
    ("visitato da Rossi in reparto", 1),
    ("con malattia di Rossi in reparto", 1),
    ("visitato da Wilson in reparto", 1),
])
def test_7_eponyms_are_explained_only_in_clinical_context(text, unexplained):
    every, left = capitalised_word_counts(text)
    assert every >= 1 and left == unexplained


def test_7_the_flag_never_changes_rejections():
    parts = [("Paziente di ", "NESSUNA"), ("52", "ETÀ"), (" anni, visitato dal Dott. Parkinson per una malattia di Parkinson. ", "NESSUNA")] + BASE[2:]
    result = validate_case(case_from(parts), spec_for(chars=len("".join(t for t, _ in parts))), context(V2))
    assert "residual_title_name" in result.reasons, "title plus name is still rejected"
    assert result.flags["midsentence_capitalised_in_redacted"] >= result.flags["capitalised_not_explained_by_eponym"] >= 1


# The whole pipeline with v2 rules

def test_v2_run_accepts_varied_notes_and_the_audit_passes(project):
    report = run_v2(project, DiverseClient())
    assert report["cases_accepted"] == report["planned_cases"] == 15 and not report["rejection_reason_counts"]
    config = json.loads((project["out"] / "fold_1/non_anchored/run_config.json").read_text(encoding="utf-8"))
    assert config["prompt_version"] == "v2" and config["template_sha256"] == V2_SHA and config["plan_details"]["reference_year_source"] in ("training_partition_dates", "configured_fallback")
    audit = audit_fold(v2_cfg(project["cfg"]), 1, "non_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert audit["passed"]


def test_v2_prompts_carry_the_age_and_year_lines(project):
    client = DiverseClient()
    run_v2(project, client)
    joined = "\n".join(client.prompts)
    assert "Età del paziente:" in joined and "Un unico paragrafo continuo" in joined


def test_6_copying_an_accepted_note_within_the_run_is_rejected(project):
    def copy_first(client, cid, case):
        if client.first is None:
            client.first = case["segments"]
            return case
        if case["segments"][1]["entity_type"] == "NESSUNA" and len(client.first) and not any(s["entity_type"] == "NOME" for s in case["segments"]):
            return case
        return {"case_id": cid, "segments": client.first[:1] + case["segments"][1:2] + client.first[1:]}
    run_v2(project, DiverseClient(copy_first), max_requests=2)
    rows = log(project)
    assert any("near_duplicate_of_accepted_note" in r["reasons"] or "exact_duplicate" in r["reasons"] for r in rows if r["status"] == "rejected")


def test_6_the_check_covers_notes_accepted_before_a_resume(project):
    """Each replayed note copies half of a note accepted before the restart, with its own distinct ending,
    so only memory carried across the resume can catch the first one."""
    run_v2(project, DiverseClient(), max_requests=1)
    first_text = json.loads((project["out"] / "fold_1/non_anchored/examples.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
    before = {r["case_id"] for r in log(project)}

    def replay(client, cid, case):
        own = " ".join(f"{w}{cid[-2:]}" for w in "fine nota diversa per ogni caso".split())
        return {"case_id": cid, "segments": [{"text": first_text[: len(first_text) // 2] + " " + own + ". ", "entity_type": "NESSUNA"}] + case["segments"][1:]}
    run_v2(project, DiverseClient(replay), max_requests=1)
    second = [r for r in log(project) if r["case_id"] not in before]
    assert second and all(r["status"] == "rejected" and "near_duplicate_of_accepted_note" in r["reasons"] for r in second)


def test_2_a_line_break_from_the_model_is_rejected_and_preserved(project):
    def add_break(client, cid, case):
        case["segments"][0]["text"] = case["segments"][0]["text"] + "\n"
        return case
    run_v2(project, DiverseClient(add_break), max_requests=1)
    rows = [r for r in log(project) if r["status"] == "rejected"]
    assert rows and all("line_break_in_text" in r["reasons"] for r in rows)
    assert (project["out"] / "fold_1/non_anchored/rejected.jsonl").exists()


def test_case_id_filter_is_for_development_only(project, monkeypatch):
    client = DiverseClient()
    run_v2(project, client, case_ids=["F1-N00000", "F1-N00004"])
    asked = {c for p in client.prompts for c in re.findall(r"SCHEDA (F\d-N\d+)", p)}
    assert asked == {"F1-N00000", "F1-N00004"}
    monkeypatch.setattr(g, "ROOT", project["tmp"])
    with pytest.raises(PermissionError, match="partial dataset"):
        g.run_generation(v2_cfg(project["cfg"]), 1, "non_anchored", execute=True, client=DiverseClient(), output_root=project["tmp"] / "synthetic_data",
                         gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"], case_ids=["F1-N00000"])


def test_both_arms_share_the_v2_plan_and_differ_only_by_examples(project):
    from src.data.splits import load_fold, load_train_notes
    from src.generation.anchor_selection import AnchorSelector
    spec = load_fold(1, project["splits"])
    train = load_train_notes(spec, project["gold"])
    cfg = v2_cfg(project["cfg"])
    plan = build_plan(1, train, cfg)
    sel = AnchorSelector(spec, train, cfg["anchors"]["strategy"], cfg["seed"], cfg["anchors"]["n"])
    group = g.group_requests(plan, 1, 3)[0]
    anchored, _ = g.build_request_prompt(group, "style_anchored", sel, cfg, project["prompts"])
    plain, _ = g.build_request_prompt(group, "non_anchored", sel, cfg, project["prompts"])
    assert anchored != plain and plain.split("SCHEDE")[1].strip() in anchored


# Owner decision 2026-11-09: an age written as one Italian number word is a valid ETÀ when accept_spelled_ages is set.
@pytest.mark.parametrize("word,value", [("settantadue", 72), ("ventuno", 21), ("trentotto", 38), ("ventitré", 23), ("diciannove", 19),
                                        ("novanta", 90), ("cento", 100), ("centouno", 101), ("centotto", 108), ("Cinquantadue", 52)])
def test_italian_number_words_are_read(word, value):
    from src.generation.synthetic_validation import italian_number_word
    assert italian_number_word(word) == value


@pytest.mark.parametrize("word", ["un", "una", "zero", "duecento", "settantaduemila", "anni", "ventiuno", "54", "cinquanta due"])
def test_other_words_are_not_ages(word):
    from src.generation.synthetic_validation import italian_number_word
    assert italian_number_word(word) is None


SPELLED = [("Paziente di ", "NESSUNA"), ("cinquantadue", "ETÀ")] + BASE[2:]


def test_spelled_age_is_accepted_and_matches_the_patient_age():
    assert V2["validation"]["accept_spelled_ages"] is True
    assert reasons(SPELLED) == []


def test_spelled_age_is_rejected_when_not_enabled():
    cfg = copy.deepcopy(V2)
    cfg["validation"]["accept_spelled_ages"] = False
    assert "eta_not_numeric" in reasons(SPELLED, cfg)


def test_a_wrong_spelled_age_does_not_count_as_the_patient_age():
    parts = [("Paziente di ", "NESSUNA"), ("quarantasette", "ETÀ")] + BASE[2:]
    assert "patient_age_missing" in reasons(parts)
