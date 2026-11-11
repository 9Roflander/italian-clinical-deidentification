import json
import subprocess
from types import SimpleNamespace

import pytest

from src.data.loaders import ROOT
from src.generation.gemini_client import GeminiClient, QuotaExhausted, RateLimiter, load_api_key
from src.generation.privacy_screening import scrub_secret
from src.generation.prompt_templates import RESPONSE_JSON_SCHEMA
from src.generation.synthetic_generator import load_config
from tests.conftest import FakeClient

KEY = "AQ.fakekeyfortestsonly0000000000000000000000000000000"


def test_key_file_wins_over_a_stale_environment_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "stale")
    f = tmp_path / "k.txt"
    f.write_text(KEY)
    assert load_api_key(f) == KEY


def test_environment_is_the_fallback_when_no_file_exists(tmp_path, monkeypatch):
    import src.generation.gemini_client as gc
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(gc, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    assert load_api_key() == KEY


def test_explicit_missing_file_is_an_error_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    with pytest.raises(FileNotFoundError):
        load_api_key(tmp_path / "missing")


def test_bare_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "k.txt"
    f.write_text(KEY + "\n")
    assert load_api_key(f) == KEY


def test_curl_snippet_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "k.txt"
    f.write_text(f'curl "https://example/models/x:generateContent" \\\n  -H \'Content-Type: application/json\' \\\n  -H \'X-goog-api-key: {KEY}\' \\\n  -X POST\n')
    assert load_api_key(f) == KEY


def test_unusable_file_errors_without_echoing_content(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "k.txt"
    f.write_text("two words secretvalue\n")
    with pytest.raises(ValueError) as e:
        load_api_key(f)
    assert "secretvalue" not in str(e.value)


def test_scrub():
    assert KEY not in scrub_secret(f"failed with {KEY}", KEY)
    assert scrub_secret("no key here", None) == "no key here"


def _client(error, retries=0):
    c = GeminiClient.__new__(GeminiClient)
    c.cfg, c.schema, c._secret = load_config(ROOT / "configs" / "generation.yaml"), RESPONSE_JSON_SCHEMA, KEY
    c.model_id, c.max_retries, c.limiter, c.max_wait_s = "m", retries, RateLimiter(0), 300
    from google.genai import types
    c._types = types

    def boom(**kw):
        raise error
    c._client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    return c


class FakeApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP failure for key {KEY}")
        self.code = code


@pytest.mark.parametrize("code", [400, 500])
def test_client_errors_never_contain_the_key(code):
    with pytest.raises(RuntimeError) as e:
        _client(FakeApiError(code)).generate("p", 1)
    assert KEY not in str(e.value) and "[REDACTED_KEY]" in str(e.value)


def test_exhausted_429_stops_the_run_without_the_key():
    with pytest.raises(QuotaExhausted) as e:
        _client(FakeApiError(429)).generate("p", 1)
    assert KEY not in str(e.value)


def test_key_file_is_ignored_by_git():
    r = subprocess.run(["git", "check-ignore", "key.txt"], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == "key.txt"


def test_secure_outputs_are_ignored_by_git():
    for path in ["data_secure/gold/gold_v1.jsonl", "data_secure/gold/annotation_review.csv", "predictions_secure/x.jsonl", "checkpoints/x", "prompts/rendered/x.txt", "synthetic_data/fold_1/style_anchored/examples.jsonl",
                 "synthetic_data/fold_1/style_anchored/raw_responses.jsonl", "synthetic_data/fold_1/style_anchored/rejected.jsonl"]:
        r = subprocess.run(["git", "check-ignore", path], cwd=ROOT, capture_output=True, text=True)
        assert r.stdout.strip() == path, path


def test_retry_delay_is_read_from_both_message_forms():
    from src.generation.gemini_client import retry_delay_seconds
    assert retry_delay_seconds("Please retry in 20h19m53.59s.") == 20 * 3600 + 19 * 60 + 53.59
    assert retry_delay_seconds("{'retryDelay': '73193s'}") == 73193
    assert retry_delay_seconds("no delay here") is None


def test_daily_quota_stops_immediately_instead_of_burning_retries():
    calls = []
    c = _client(FakeApiError(429), retries=5)

    def boom(**kw):
        calls.append(1)
        e = FakeApiError(429)
        e.args = ("Please retry in 20h19m53s.",)
        raise e
    from types import SimpleNamespace
    c._client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    with pytest.raises(QuotaExhausted):
        c.generate("p", 1)
    assert len(calls) == 1
