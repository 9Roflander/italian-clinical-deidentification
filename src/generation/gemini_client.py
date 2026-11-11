"""Thin Gemini client. The API key is never logged, and errors pass through scrub_secret."""
from __future__ import annotations

import os
import pathlib
import random
import re
import threading
import time
from dataclasses import dataclass

from src.generation.privacy_screening import scrub_secret


class QuotaExhausted(RuntimeError):
    """Stop the run. Everything already saved is kept and the run can resume."""


def load_api_key(path=None) -> str:
    """Order: explicit path, then key.txt in the project root, then GEMINI_API_KEY.

    The project file wins over the environment, because a stale key in a shell profile
    would otherwise silently replace the key chosen for this project.
    """
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parents[2] / "key.txt"
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        m = re.search(r"x-goog-api-key:\s*([^'\"\s]+)", txt, re.I)
        if m:
            return m.group(1)
        if len(txt.split()) == 1:
            return txt
        raise ValueError("could not find an API key in the key file")
    if path:
        raise FileNotFoundError("the given key file does not exist")
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    raise FileNotFoundError("no API key: provide key.txt or set GEMINI_API_KEY")


@dataclass
class GenerationResult:
    text: str | None
    finish_reason: str | None
    model_version: str | None
    response_id: str | None
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    total_tokens: int
    latency_s: float
    prompt_blocked: str | None = None


def retry_delay_seconds(message: str) -> float | None:
    """The delay the API asks for on a 429, if it states one."""
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", message)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", message)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    return None


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        if at > now:
            time.sleep(at - now)


class GeminiClient:
    def __init__(self, api_key: str, cfg: dict, schema: dict):
        from google import genai
        from google.genai import types
        self._types, self._secret, self.cfg, self.schema = types, api_key, cfg, schema
        self.model_id = cfg["model"]["id"]
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(cfg["model"]["timeout_s"] * 1000)))
        self.limiter = RateLimiter(cfg["limits"]["requests_per_minute"])
        self.max_retries = cfg["limits"]["max_retries"]
        self.max_wait_s = cfg["limits"].get("max_retry_wait_s", 300)
        self.sdk_version = genai.__version__

    def model_info(self) -> dict:
        """What the API reports about the model at run start; recorded for provenance."""
        try:
            m = self._client.models.get(model=self.model_id)
            return {"name": m.name, "version": m.version, "display_name": m.display_name}
        except Exception as e:
            return {"error": scrub_secret(type(e).__name__, self._secret)}

    def generate(self, prompt: str, seed: int) -> GenerationResult:
        m, t = self.cfg["model"], self._types
        config = t.GenerateContentConfig(
            temperature=m["temperature"], top_p=m["top_p"], max_output_tokens=m["max_output_tokens"], seed=seed,
            response_mime_type="application/json", response_json_schema=self.schema,
            thinking_config=t.ThinkingConfig(thinking_level=m["thinking_level"]))
        last = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            start = time.monotonic()
            try:
                r = self._client.models.generate_content(model=self.model_id, contents=prompt, config=config)
                return self._to_result(r, time.monotonic() - start)
            except Exception as e:  # SDK raises several error classes; classify by status code
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                last = scrub_secret(f"{type(e).__name__} {code}: {e}", self._secret)[:1200]
                if code in (400, 401, 403, 404):
                    raise RuntimeError(last) from None
                wait = retry_delay_seconds(last) if code == 429 else None
                # Failed attempts can count against daily quota, so do not retry a long wait.
                if code == 429 and wait is not None and wait > self.max_wait_s:
                    raise QuotaExhausted(last) from None
                if attempt == self.max_retries:
                    if code == 429:
                        raise QuotaExhausted(last) from None
                    raise RuntimeError(last) from None
                time.sleep(wait if wait is not None else min(120, 2 ** attempt * 5) + random.uniform(0, 2))
        raise RuntimeError(last)

    @staticmethod
    def _to_result(r, latency: float) -> GenerationResult:
        u = r.usage_metadata
        cand = r.candidates[0] if r.candidates else None
        finish = str(cand.finish_reason.value if cand and cand.finish_reason else None)
        blocked = str(r.prompt_feedback.block_reason) if r.prompt_feedback and r.prompt_feedback.block_reason else None
        try:
            text = r.text
        except Exception:
            text = None
        return GenerationResult(text, finish, r.model_version, r.response_id,
                                (u.prompt_token_count or 0) if u else 0, (u.candidates_token_count or 0) if u else 0,
                                (u.thoughts_token_count or 0) if u else 0, (u.total_token_count or 0) if u else 0,
                                round(latency, 2), blocked)
