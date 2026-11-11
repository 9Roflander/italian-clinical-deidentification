"""Versioned prompt rendering.

Both arms use the same template, so a rendered non-anchored prompt equals the rendered
anchored prompt with the anchor block removed. That is the controlled difference the
style-anchoring ablation relies on.
"""
from __future__ import annotations

import hashlib
import pathlib

from src.data.loaders import ROOT

PROMPTS_DIR = ROOT / "prompts"
SEGMENT_TYPES = ["NESSUNA", "NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"cases": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "segments": {"type": "array", "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "entity_type": {"type": "string", "enum": SEGMENT_TYPES}},
                "required": ["text", "entity_type"]}}},
        "required": ["case_id", "segments"]}}},
    "required": ["cases"],
}


def read_template(version: str, prompts_dir=PROMPTS_DIR) -> tuple[str, str]:
    d = pathlib.Path(prompts_dir)
    return ((d / f"generation_{version}.txt").read_text(encoding="utf-8"),
            (d / f"anchor_block_{version}.txt").read_text(encoding="utf-8"))


def template_sha(version: str, prompts_dir=PROMPTS_DIR) -> str:
    main, block = read_template(version, prompts_dir)
    return hashlib.sha256((main + "\0" + block).encode("utf-8")).hexdigest()


def render_case_spec(spec) -> str:
    lines = [f"SCHEDA {spec.case_id}", f"- Specialità: {spec.specialty}",
             f"- Paziente: {spec.gender}, fascia d'età {spec.age_band} anni", f"- Lunghezza: circa {spec.target_chars} caratteri"]
    if spec.record_type == "synthetic_negative":
        lines.append("- Occorrenze richieste: NOME 0, ETÀ 0, DATA 0, LUOGO/INDIRIZZO 0. Nessuna informazione sensibile nel testo.")
        return "\n".join(lines)
    lines.append(f"- Occorrenze richieste: NOME {spec.n_nome}, ETÀ {spec.n_eta}, DATA {spec.n_data}, LUOGO/INDIRIZZO {spec.n_luogo}")
    if getattr(spec, "patient_age", None) is not None:
        lines.append(f"- Età del paziente: {spec.patient_age} anni")
    if getattr(spec, "reference_year", None) is not None:
        lines.append(f"- Anno di riferimento: {spec.reference_year}")
    if spec.n_luogo:
        lines.append(f"- Area geografica: {spec.area}")
    if spec.n_nome:
        lines.append(f"- Forma del nome: {spec.name_form}. Il nome del paziente inizia con \"{spec.name_letter}\"; gli altri nomi usano iniziali diverse")
    return "\n".join(lines)


def render_prompt(version: str, specs, anchors=(), prompts_dir=PROMPTS_DIR) -> str:
    """`anchors` are already-masked (or explicitly approved raw) texts, in prompt order."""
    main, block = read_template(version, prompts_dir)
    anchor_block = ""
    if anchors:
        body = "".join(f"\n--- ESEMPIO {i} ---\n{text}\n" for i, text in enumerate(anchors, 1))
        anchor_block = block.replace("{{ANCHORS}}", body)
    out = (main.replace("{{N_CASES}}", str(len(specs)))
               .replace("{{ANCHOR_BLOCK}}", anchor_block)
               .replace("{{CASE_SPECS}}", "\n\n".join(render_case_spec(s) for s in specs)))
    if "{{" in out:
        raise ValueError("unfilled template field")
    return out


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
