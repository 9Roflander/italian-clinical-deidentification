"""Fold-specific synthetic note generation.

Dry run is the default and makes no API calls. Real calls need --execute.
The anchored arm sends text derived from gold notes, so a real run also needs
--confirm-anchors-approved. Raw mode additionally needs --confirm-raw-anchors-authorized.

    python -m src.generation.synthetic_generator --fold 1 --arm non_anchored
    python -m src.generation.synthetic_generator --fold 1 --arm style_anchored --execute --confirm-anchors-approved --max-requests 5 --output-root experiments/development/test

Output under synthetic_data/ is official and needs committed source. Output elsewhere is recorded as a development run.

Runs are resumable: finished cases are skipped, and every attempt stays on disk.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading

import yaml

from src.data.loaders import DEFAULT_GOLD, ROOT
from src.data.normalization import doc_hash, tokens, word_ngrams
from src.data.splits import SPLITS_DIR, LeakageError, load_fold, load_train_notes
from src.infrastructure import INFRASTRUCTURE_VERSION
from src.infrastructure.provenance import check_source_integrity, is_official_output, run_environment
from src.generation.anchor_selection import AnchorSelector
from src.generation.gemini_client import GeminiClient, QuotaExhausted, load_api_key
from src.generation.generation_plan import build_plan, plan_details, plan_sha, plan_summary, write_plan
from src.generation.privacy_screening import count_identifying_strings, mask_anchor_set
from src.generation.prompt_templates import PROMPTS_DIR, RESPONSE_JSON_SCHEMA, prompt_sha, render_prompt, template_sha
from src.generation.synthetic_validation import ValidationContext, build_context, validate_case

ARMS = ("style_anchored", "non_anchored")


def load_config(path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def request_seed(base: int, fold: int, index: int, attempt: int) -> int:
    """Shared by both arms on purpose: arms differ by anchors, not by seed."""
    digest = hashlib.sha256(f"{base}|{fold}|{index}|{attempt}".encode()).hexdigest()
    return int(digest[:8], 16) % (2 ** 31 - 1)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def git_is_dirty() -> bool:
    """True when experiment-critical source differs from the recorded commit. A run's own outputs do not count."""
    return not check_source_integrity(ROOT).clean


def group_requests(specs, attempt: int, size: int) -> list[dict]:
    """Stable request IDs, so a resumed run produces the same requests and seeds."""
    groups = []
    for i in range(0, len(specs), size):
        chunk = specs[i : i + size]
        index = chunk[0].order_index // size + (attempt - 1) * 1009
        groups.append({"index": index, "attempt": attempt, "specs": chunk})
    return groups


def build_request_prompt(group, arm, selector, cfg, prompts_dir) -> tuple[str, list]:
    """Render one request. Returns the prompt and the anchor notes used."""
    if arm == "non_anchored":
        return render_prompt(cfg["prompt_version"], group["specs"], (), prompts_dir), []
    anchors = selector.for_request(group["index"])
    mode = cfg["anchors"]["mode"]
    if mode == "masked":
        texts = [m.text for m in mask_anchor_set(anchors)]
        if count_identifying_strings("\n".join(texts), anchors):
            raise LeakageError("masked anchor text still contains an annotated name or place")
    elif mode == "raw":
        texts = [a.text for a in anchors]
    else:
        raise ValueError("anchors.mode must be masked or raw")
    return render_prompt(cfg["prompt_version"], group["specs"], texts, prompts_dir), anchors


def estimate_cost(requests: int, cases, prompt_chars: int, cfg: dict) -> dict:
    """Rough estimate. Replace with measured tokens per request after a smoke test."""
    p = cfg["pricing"]
    prompt_tokens = prompt_chars / 3.5
    output_tokens = sum(s.target_chars / 3.2 + 12 * (2 * sum(s.counts.values()) + 1) for s in cases)
    thinking = cfg.get("estimate", {}).get("thought_tokens_per_request", 300) * requests
    cost = prompt_tokens / 1e6 * p["input_per_million"] + (output_tokens + thinking) / 1e6 * p["output_per_million"]
    return {"requests": requests, "prompt_tokens_est": int(prompt_tokens), "output_tokens_est": int(output_tokens + thinking),
            "usd_first_pass_est": round(cost, 2), "assumes": "chars/3.5 input, chars/3.2 output, no rejections"}


class Run:
    """Holds output paths and append-only logs for one fold and arm."""

    def __init__(self, out_dir: pathlib.Path):
        self.dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, name: str) -> pathlib.Path:
        return self.dir / name

    def append(self, name: str, row: dict) -> None:
        with self.lock:
            with open(self.path(name), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def read(self, name: str) -> list[dict]:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def process_response(result, group, arm, anchors, prompt, seed, run: Run, ctx, cfg, base_meta) -> None:
    """Log the request, validate each case, save accepted and rejected records."""
    specs = {s.case_id: s for s in group["specs"]}
    request_id = f"F{specs[next(iter(specs))].fold}-{arm}-A{group['attempt']}-R{group['index']:05d}"
    parsed, parse_error = None, None
    if result.text:
        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError as e:
            parse_error = f"json_error at char {e.pos}"
    run.append("raw_responses.jsonl", {"request_id": request_id, "raw_text": result.text})
    run.append("requests.jsonl", {
        "request_id": request_id, "attempt": group["attempt"], "index": group["index"], "case_ids": list(specs),
        "anchor_ids": [a.note_id for a in anchors], "anchor_mode": cfg["anchors"]["mode"] if anchors else "none",
        "prompt_sha256": prompt_sha(prompt), "seed": seed, "model_id": cfg["model"]["id"], "model_version": result.model_version,
        "response_id": result.response_id, "finish_reason": result.finish_reason, "prompt_blocked": result.prompt_blocked,
        "prompt_tokens": result.prompt_tokens, "output_tokens": result.output_tokens, "thought_tokens": result.thought_tokens,
        "total_tokens": result.total_tokens, "latency_s": result.latency_s, "parse_error": parse_error, "timestamp": now()})
    returned = {}
    if isinstance(parsed, dict) and isinstance(parsed.get("cases"), list):
        for c in parsed["cases"]:
            if isinstance(c, dict):
                # Tolerate the model echoing the spec heading; the ID must still match a requested case.
                cid = str(c.get("case_id", "")).replace("SCHEDA", "").strip()
                if cid in specs:
                    returned.setdefault(cid, {**c, "case_id": cid})
    for cid, spec in specs.items():
        common = {"case_id": cid, "request_id": request_id, "attempt": group["attempt"], "record_type": spec.record_type,
                  "anchor_ids": [a.note_id for a in anchors], "arm": arm, "fold": spec.fold}
        case = returned.get(cid)
        if case is None:
            why = ("output_truncated" if str(result.finish_reason).upper().endswith("MAX_TOKENS") else
                   "prompt_blocked" if result.prompt_blocked else "json_parse_error" if parse_error else
                   "no_response_text" if not result.text else "case_missing_from_response")
            run.append("case_log.jsonl", {**common, "status": "rejected", "reasons": [why]})
            continue
        with ctx["seen_lock"]:
            ctx_now = ValidationContext(ctx["train_hashes"], ctx["train_token_docs"], ctx["train_index"], anchors, ctx["seen_hashes"], cfg,
                                        ctx["accepted_tokens"], ctx["accepted_index"])
            res = validate_case(case, spec, ctx_now)
            if res.accepted:
                ctx["seen_hashes"].add(doc_hash(res.note["text"]))
                remember_accepted(ctx, cid, res.note["text"])
        run.append("case_log.jsonl", {**common, "status": "accepted" if res.accepted else "rejected", "reasons": res.reasons, "flags": res.flags,
                                      "chars": len(res.note["text"]) if res.note else 0,
                                      "entity_counts": dict(collections.Counter(e["type"] for e in res.note["entities"])) if res.note else {}})
        if res.accepted:
            run.append("examples.jsonl", {
                "record_id": f"syn_f{spec.fold}_{arm}_{spec.order_index:05d}", "source_dataset": base_meta["dataset_name"],
                "source_record_id": cid, "record_type": spec.record_type, "contains_target_pii": spec.record_type != "synthetic_negative",
                "entity_annotations_available": True, "exact_duplicate_status": "none_in_training_partition", "near_duplicate_status": "none_in_training_partition",
                "permitted_for_training": False, "fold_id": spec.fold, "partition": "train", "generation_prompt_version": cfg["prompt_version"],
                "generation_model_version": result.model_version, "request_id": request_id, "anchor_ids": [a.note_id for a in anchors],
                "type_order": spec.type_order, "specialty": spec.specialty, "gender": spec.gender, "age_band": spec.age_band,
                "text": res.note["text"], "redacted_text": res.note["redacted_text"], "entities": res.note["entities"]})
        else:
            run.append("rejected.jsonl", {"case_id": cid, "request_id": request_id, "reasons": res.reasons, "note": res.note})


def remember_accepted(ctx: dict, key: str, text: str) -> None:
    """Index an accepted note, so later notes in this fold and arm can be checked against it."""
    toks = tokens(text)
    ctx["accepted_tokens"][key] = toks
    for g in word_ngrams(toks, 8):
        ctx["accepted_index"].setdefault(g, set()).add(key)


def write_reports(run: Run, plan, arm, meta) -> dict:
    logs = run.read("case_log.jsonl")
    reqs = run.read("requests.jsonl")
    last = {}
    for row in logs:
        last[row["case_id"]] = row
    reasons, accepted_by_type, actual = collections.Counter(), collections.Counter(), collections.Counter()
    accepted_chars = []
    for row in logs:
        reasons.update(row.get("reasons", []))
    for row in last.values():
        if row["status"] == "accepted":
            accepted_by_type[row["record_type"]] += 1
            actual.update(row.get("entity_counts", {}))
            accepted_chars.append(row.get("chars", 0))
    p = meta["cfg"]["pricing"]
    prompt_tok, out_tok, thought_tok = (sum(r.get(k, 0) for r in reqs) for k in ("prompt_tokens", "output_tokens", "thought_tokens"))
    cost = prompt_tok / 1e6 * p["input_per_million"] + (out_tok + thought_tok) / 1e6 * p["output_per_million"]
    report = {
        "fold": meta["fold"], "arm": arm, "planned_cases": len(plan), "cases_attempted": len(last),
        "cases_accepted": sum(accepted_by_type.values()), "attempts_logged": len(logs), "accepted_by_type": dict(accepted_by_type),
        "rejection_reason_counts": dict(reasons.most_common()), "requests": len(reqs),
        "tokens": {"prompt": prompt_tok, "output": out_tok, "thoughts": thought_tok}, "cost_usd_at_configured_price": round(cost, 4),
        "models_returned": sorted({str(r.get("model_version")) for r in reqs}),
        "finish_reasons": dict(collections.Counter(str(r.get("finish_reason")) for r in reqs)),
        "accepted_mentions_per_note": {k: round(v / max(1, sum(accepted_by_type.values())), 2) for k, v in actual.items()},
        "accepted_median_chars": sorted(accepted_chars)[len(accepted_chars) // 2] if accepted_chars else None,
        "acceptance_rate_per_attempt": round(sum(1 for r in logs if r["status"] == "accepted") / max(1, len(logs)), 3),
        "post_audit": "not_run", "permitted_for_training": False}
    run.path("validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    fields = ["record_id", "case_id", "record_type", "fold", "arm", "status", "attempts", "final_reasons", "anchor_ids", "request_id", "generation_prompt_version",
              "generation_model_version", "permitted_for_training"]
    versions = {r["request_id"]: r.get("model_version") for r in reqs}
    attempts = collections.Counter(r["case_id"] for r in logs)
    with open(run.path("generation_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in plan:
            row = last.get(s.case_id)
            w.writerow({"record_id": f"syn_f{s.fold}_{arm}_{s.order_index:05d}", "case_id": s.case_id, "record_type": s.record_type,
                        "fold": s.fold, "arm": arm, "status": row["status"] if row else "not_attempted", "attempts": attempts.get(s.case_id, 0),
                        "final_reasons": ";".join(row.get("reasons", [])) if row else "", "anchor_ids": ";".join(row.get("anchor_ids", [])) if row else "",
                        "request_id": row.get("request_id", "") if row else "", "generation_prompt_version": meta["cfg"]["prompt_version"],
                        "generation_model_version": versions.get(row.get("request_id"), "") if row else "", "permitted_for_training": False})
    return report


def run_generation(cfg: dict, fold: int, arm: str, execute: bool = False, max_requests: int | None = None, output_root=None,
                   confirm_raw: bool = False, confirm_anchors: bool = False, require_clean_tree: bool = False, client=None, gold_path=DEFAULT_GOLD, splits_dir=SPLITS_DIR, prompts_dir=PROMPTS_DIR,
                   key_file=None, write_sample_prompts: int = 0, record_types=None, case_ids=None) -> dict:
    """`record_types` and `case_ids` limit a development pilot. Official runs always cover the whole plan."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    out_root = pathlib.Path(output_root) if output_root else ROOT / "synthetic_data"
    official = is_official_output(out_root, ROOT)
    integrity = check_source_integrity(ROOT)
    if record_types is not None:
        unknown = set(record_types) - {"synthetic_named", "synthetic_name_free", "synthetic_negative"}
        if unknown or not record_types:
            raise ValueError(f"unknown record types: {sorted(unknown)}")
        if official:
            raise PermissionError("a record-type filter makes a partial dataset, so it is only allowed outside the official folders")
    if case_ids is not None and official:
        raise PermissionError("a case-ID filter makes a partial dataset, so it is only allowed outside the official folders")
    if execute and (require_clean_tree or official) and git_is_dirty():
        raise PermissionError("official runs need committed source, so the run can be tied to an exact version. "
                              f"Commit first, or use --output-root outside the official folders for a development run. {integrity.summary()}")
    if execute and arm == "style_anchored" and not confirm_anchors:
        raise PermissionError("the anchored arm sends text derived from gold notes to the API; the project owner must approve this "
                              "and confirm the billing tier, then pass --confirm-anchors-approved")
    if arm == "style_anchored" and cfg["anchors"]["mode"] == "raw" and not confirm_raw:
        raise PermissionError("raw anchors send unmasked notes to the API; pass --confirm-raw-anchors-authorized")
    spec = load_fold(fold, splits_dir)
    train_notes = load_train_notes(spec, gold_path)      # test-note text never enters this process
    selector = AnchorSelector(spec, train_notes, cfg["anchors"]["strategy"], cfg["seed"], cfg["anchors"]["n"])
    plan = build_plan(fold, train_notes, cfg)
    run = Run(out_root / f"fold_{fold}" / arm)
    size = cfg["plan"]["cases_per_request"]
    meta = {"fold": fold, "arm": arm, "cfg": cfg, "dataset_name": f"gemini_generated_{cfg['generation_version']}"}
    run_config = {"fold": fold, "arm": arm, "generation_version": cfg["generation_version"], "prompt_version": cfg["prompt_version"],
                  "template_sha256": template_sha(cfg["prompt_version"], prompts_dir), "plan_sha256": plan_sha(plan), "gold_sha256": spec.gold_sha256,
                  "git_commit": git_commit(), "git_dirty": not integrity.clean, "source_integrity": integrity.to_manifest(), "official_run": official,
                  "infrastructure_version": INFRASTRUCTURE_VERSION, "environment": run_environment(), "anchor_pool_size": len(selector.pool_ids), "anchors_excluded_from_pool": selector.excluded_ids,
                  "model": cfg["model"], "anchors": cfg["anchors"] if arm == "style_anchored" else {"mode": "none"}, "record_type_filter": sorted(record_types) if record_types else None, "case_id_filter": sorted(case_ids) if case_ids else None,
                  "plan_details": plan_details(train_notes, cfg),
                  "validation": cfg["validation"], "limits": cfg["limits"], "started": now()}

    # Render everything first: a guard failure must stop the run before any API call.
    first = group_requests(plan, 1, size)
    prompt_chars = 0
    anchors_used = collections.Counter()
    samples = []
    for g in first:
        prompt, anchors = build_request_prompt(g, arm, selector, cfg, prompts_dir)
        prompt_chars += len(prompt)
        anchors_used.update(a.note_id for a in anchors)
        if len(samples) < write_sample_prompts:
            samples.append((g["index"], prompt))
    if samples:
        rendered = ROOT / "prompts" / "rendered"
        rendered.mkdir(parents=True, exist_ok=True)
        for idx, text in samples:
            (rendered / f"fold{fold}_{arm}_R{idx:05d}.txt").write_text(text, encoding="utf-8")
    summary = {"fold": fold, "arm": arm, "plan": plan_summary(plan), "plan_sha256": plan_sha(plan), "requests_first_pass": len(first),
               "mean_prompt_chars": int(prompt_chars / max(1, len(first))), "distinct_anchors_used": len(anchors_used),
               "anchor_use_min_max": [min(anchors_used.values()), max(anchors_used.values())] if anchors_used else [0, 0],
               "anchor_pool_size": len(selector.pool_ids), "anchors_excluded_from_pool": selector.excluded_ids,
               "cost_estimate": estimate_cost(len(first), plan, prompt_chars, cfg), "guards_passed_on_all_prompts": True}
    if not execute:
        run.path("dry_run_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["mode"] = "dry_run"
        return summary

    write_plan(plan, run.path("generation_plan.csv"))
    if client is None:
        client = GeminiClient(load_api_key(key_file), cfg, RESPONSE_JSON_SCHEMA)
    run_config["sdk_version"] = getattr(client, "sdk_version", "unknown")
    run_config["model_reported_by_api"] = client.model_info() if hasattr(client, "model_info") else {}
    if run.path("run_config.json").exists():
        earlier = json.loads(run.path("run_config.json").read_text(encoding="utf-8")).get("source_integrity") or {}
        tree, previous = integrity.critical_tree_sha256, earlier.get("critical_tree_sha256")
        if official and previous and previous != tree:
            raise PermissionError("this run started from different committed source, so resuming would mix two versions in one dataset. "
                                  "Use a new output folder, or return to the original commit")
    run.append("run_config_history.jsonl", json.loads(json.dumps(run_config, default=str)))
    run.path("run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ctx = build_context(train_notes, cfg)
    ctx["seen_hashes"] = set()
    ctx["seen_lock"] = threading.Lock()
    ctx["accepted_tokens"], ctx["accepted_index"] = {}, {}
    for row in run.read("examples.jsonl"):
        ctx["seen_hashes"].add(doc_hash(row["text"]))
        remember_accepted(ctx, row["source_record_id"], row["text"])

    spent = lambda: sum(r.get("prompt_tokens", 0) / 1e6 * cfg["pricing"]["input_per_million"]
                        + (r.get("output_tokens", 0) + r.get("thought_tokens", 0)) / 1e6 * cfg["pricing"]["output_per_million"]
                        for r in run.read("requests.jsonl"))
    made = 0
    stop = threading.Event()
    stop_reason = None

    def call(group):
        nonlocal stop_reason
        if stop.is_set():
            return
        prompt, anchors = build_request_prompt(group, arm, selector, cfg, prompts_dir)
        seed = request_seed(cfg["seed"], fold, group["index"], group["attempt"])
        try:
            result = client.generate(prompt, seed)
        except QuotaExhausted as e:
            run.append("errors.jsonl", {"timestamp": now(), "index": group["index"], "attempt": group["attempt"], "kind": "quota", "error": str(e)})
            stop_reason = "quota exhausted, see errors.jsonl; rerun to resume after it resets"
            stop.set()
            return
        except RuntimeError as e:
            run.append("errors.jsonl", {"timestamp": now(), "index": group["index"], "attempt": group["attempt"], "error": str(e)})
            stop_reason = "api_error, see errors.jsonl"
            stop.set()
            return
        process_response(result, group, arm, anchors, prompt, seed, run, ctx, cfg, meta)

    for attempt in range(1, cfg["validation"]["max_attempts_per_case"] + 1):
        logs = run.read("case_log.jsonl")
        accepted = {r["case_id"] for r in logs if r["status"] == "accepted"}
        tried = collections.Counter(r["case_id"] for r in logs)
        pending = [s for s in plan if s.case_id not in accepted and tried[s.case_id] < attempt and (record_types is None or s.record_type in record_types)
                   and (case_ids is None or s.case_id in set(case_ids))]
        if not pending:
            continue
        groups = group_requests(pending, attempt, size)
        if max_requests is not None:
            groups = groups[: max(0, max_requests - made)]
        if not groups:
            break
        batch = cfg["limits"]["workers"] * 4
        with cf.ThreadPoolExecutor(max_workers=cfg["limits"]["workers"]) as pool:
            for i in range(0, len(groups), batch):
                if spent() >= cfg["limits"]["max_cost_usd"]:
                    stop_reason = "max_cost_usd reached"
                    stop.set()
                if stop.is_set():
                    break
                list(pool.map(call, groups[i : i + batch]))
                made += len(groups[i : i + batch])
        if stop.is_set():
            break
    report = write_reports(run, plan, arm, meta)
    report["mode"] = "executed"
    report["stop_reason"] = stop_reason
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "generation_v2.yaml"))
    ap.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--execute", action="store_true", help="make real API calls; the default is a dry run")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--output-root", help="default: synthetic_data/")
    ap.add_argument("--key-file")
    ap.add_argument("--confirm-anchors-approved", action="store_true", help="owner approved sending masked gold-derived text and confirmed the billing tier")
    ap.add_argument("--confirm-raw-anchors-authorized", action="store_true")
    ap.add_argument("--model", help="override model.id; for calibration only, with --output-root outside synthetic_data/")
    ap.add_argument("--write-sample-prompts", type=int, default=0, help="save N rendered prompts to prompts/rendered/ (git-ignored)")
    ap.add_argument("--record-type", action="append", choices=["synthetic_named", "synthetic_name_free", "synthetic_negative"], help="development pilots only")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.model:
        if not a.output_root:
            ap.error("--model requires --output-root, so calibration output cannot mix with real data")
        cfg["model"]["id"] = a.model
    out = run_generation(cfg, a.fold, a.arm, a.execute, a.max_requests, a.output_root, a.confirm_raw_anchors_authorized,
                         a.confirm_anchors_approved,
                         key_file=a.key_file, write_sample_prompts=a.write_sample_prompts, record_types=a.record_type)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("stop_reason") else 0


if __name__ == "__main__":
    sys.exit(main())
