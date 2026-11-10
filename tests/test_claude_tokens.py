"""Tests for claude_tokens.py.

Most tests use a fake client and run offline. Tests marked `api` call the real Anthropic
API (count_tokens) and are skipped when ANTHROPIC_API_KEY is not set.
"""
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import claude_tokens as ct
import experiment1_tokenizer as exp

PROJECT_DIR = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-fake-key-for-tests-0123456789"


class FakeClient:
    """Stands in for anthropic.Anthropic: tokens = 3 per word, records every call."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.messages = self

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        text = kwargs["messages"][0]["content"]
        return SimpleNamespace(input_tokens=3 * len(text.split()))


def api_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")
    response = httpx2.Response(status, request=request)
    return cls("boom", response=response, body=None)


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv(ct.KEY_VARIABLE, raising=False)


# --- API key handling -------------------------------------------------------

def test_get_api_key_reads_the_environment():
    assert ct.get_api_key({ct.KEY_VARIABLE: FAKE_KEY}) == FAKE_KEY


def test_get_api_key_strips_whitespace():
    assert ct.get_api_key({ct.KEY_VARIABLE: f"  {FAKE_KEY}\n"}) == FAKE_KEY


@pytest.mark.parametrize("environ", [{}, {ct.KEY_VARIABLE: ""}, {ct.KEY_VARIABLE: "   "}])
def test_missing_or_blank_key_raises_clear_error(environ):
    with pytest.raises(ct.MissingApiKeyError) as info:
        ct.get_api_key(environ)
    assert ct.KEY_VARIABLE in str(info.value)
    assert ".env.example" in str(info.value)


def test_make_client_uses_only_the_environment_key():
    client = ct.make_client({ct.KEY_VARIABLE: FAKE_KEY})
    assert isinstance(client, anthropic.Anthropic)
    assert client.api_key == FAKE_KEY


def test_make_client_without_key_raises_instead_of_using_other_credentials():
    with pytest.raises(ct.MissingApiKeyError):
        ct.make_client({"ANTHROPIC_AUTH_TOKEN": "something-else"})


def test_creating_a_counter_does_not_need_a_key(no_key):
    counter = ct.ClaudeTokenCounter("claude-haiku-5-5")  # lazy client
    assert "claude-haiku-5-5" in repr(counter)


def test_counter_without_key_fails_with_missing_key_error_on_use(no_key):
    with pytest.raises(ct.MissingApiKeyError):
        ct.ClaudeTokenCounter("claude-haiku-5-5").count("hello")


def test_key_is_not_in_repr_or_error_messages(monkeypatch):
    monkeypatch.setenv(ct.KEY_VARIABLE, FAKE_KEY)
    counter = ct.ClaudeTokenCounter("claude-haiku-5-5")
    _ = counter.client
    assert FAKE_KEY not in repr(counter)
    err = api_error(anthropic.AuthenticationError, 401)
    assert FAKE_KEY not in ct.describe_api_error(err)


# --- counting (fake client) ------------------------------------------------

def test_count_sends_model_and_one_user_message():
    fake = FakeClient()
    n = ct.ClaudeTokenCounter("claude-sonnet-5-5", fake).count("one two three")
    assert n == 9
    assert fake.calls == [{
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "one two three"}],
    }]


@pytest.mark.parametrize("text", ["", " ", "\n\t "])
def test_blank_text_is_zero_tokens_without_an_api_call(text):
    fake = FakeClient()
    assert ct.ClaudeTokenCounter("claude-haiku-5-5", fake).count(text) == 0
    assert fake.calls == []


@pytest.mark.parametrize("bad", [None, 123, b"bytes", ["a"]])
def test_non_string_text_raises_type_error(bad):
    with pytest.raises(TypeError):
        ct.ClaudeTokenCounter("claude-haiku-5-5", FakeClient()).count(bad)


def test_unicode_text_is_passed_unchanged():
    fake = FakeClient()
    text = "Привет 👋 мир"
    ct.ClaudeTokenCounter("claude-opus-5-5", fake).count(text)
    assert fake.calls[0]["messages"][0]["content"] == text


def test_default_models_are_the_current_claude_models():
    assert ct.CLAUDE_MODELS == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]


def test_count_corpus_has_the_same_shape_as_experiment_one():
    res = ct.count_corpus(ct.ClaudeTokenCounter("claude-haiku-5-5", FakeClient()))
    assert set(res) == {"languages", "ratios"}
    assert set(res["languages"]) == set(exp.CORPUS)
    ru = res["languages"]["ru"]
    assert ru["n_texts"] == len(exp.CORPUS["ru"])
    assert ru["total_tokens"] == 3 * ru["total_words"]  # fake: 3 tokens per word
    assert set(res["ratios"]) == {key for key, _, _ in exp.RATIOS}


# --- command line without a key / with API errors ---------------------------

def run_main(argv, capsys, **kwargs):
    code = ct.main(argv, load_dotenv=False, **kwargs)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_without_key_prints_instructions_and_does_not_crash(capsys):
    code, out, err = run_main(["Привет"], capsys, environ={})
    assert code == ct.EXIT_NO_KEY
    assert ct.KEY_VARIABLE in err
    assert "Traceback" not in err and out == ""


def test_cli_prints_a_count_per_model(capsys):
    code, out, _ = run_main(["a b"], capsys, client=FakeClient())
    assert code == ct.EXIT_OK
    for model in ct.CLAUDE_MODELS:
        assert f"{model}: 6 токенов" in out


def test_cli_model_option_limits_the_models(capsys):
    code, out, _ = run_main(["--model", "claude-haiku-5-5", "a"], capsys, client=FakeClient())
    assert code == ct.EXIT_OK
    assert "claude-haiku-5-5" in out and "claude-opus-5-5" not in out


def test_cli_needs_text_or_corpus(capsys):
    code, _, err = run_main([], capsys, client=FakeClient())
    assert code != ct.EXIT_OK and "--corpus" in err
    code, _, _ = run_main(["text", "--corpus"], capsys, client=FakeClient())
    assert code != ct.EXIT_OK


def test_cli_corpus_saves_results(capsys, tmp_path, monkeypatch):
    target = tmp_path / "claude.json"
    monkeypatch.setattr(ct, "CLAUDE_RESULTS_FILE", target)
    code, out, _ = run_main(["--corpus", "--model", "claude-haiku-5-5"], capsys, client=FakeClient())
    assert code == ct.EXIT_OK
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert list(saved) == ["claude-haiku-5-5"]
    assert saved["claude-haiku-5-5"]["languages"]["en"]["n_texts"] == len(exp.CORPUS["en"])


@pytest.mark.parametrize("cls,status,fragment", [
    (anthropic.AuthenticationError, 401, "отклонён"),
    (anthropic.NotFoundError, 404, "Модель не найдена"),
    (anthropic.RateLimitError, 429, "лимит"),
    (anthropic.PermissionDeniedError, 403, "доступа"),
])
def test_cli_turns_api_errors_into_messages(capsys, cls, status, fragment):
    code, _, err = run_main(["hello"], capsys, client=FakeClient(error=api_error(cls, status)))
    assert code == ct.EXIT_API_ERROR
    assert fragment in err and "Traceback" not in err


def test_cli_connection_error_message(capsys):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")
    code, _, err = run_main(["hello"], capsys, client=FakeClient(error=anthropic.APIConnectionError(request=request)))
    assert code == ct.EXIT_API_ERROR and "соединения" in err


# --- the key never ends up in the repository --------------------------------

def test_env_files_are_ignored_by_git_except_the_example():
    lines = (PROJECT_DIR / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in lines and "!.env.example" in lines


def test_env_example_has_the_variable_but_no_key():
    values = dict(
        line.split("=", 1)
        for line in (PROJECT_DIR / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert values == {ct.KEY_VARIABLE: ""}


def test_no_api_key_pattern_in_tracked_or_new_files():
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")
    for name in filter(None, files):
        path = PROJECT_DIR / name
        if path.is_file() and path.suffix in {".py", ".md", ".json", ".txt", ".ini", ".example", ""}:
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), name


# --- real API (skipped without a key) --------------------------------------

requires_key = pytest.mark.skipif(
    not os.environ.get(ct.KEY_VARIABLE, "").strip(),
    reason=f"{ct.KEY_VARIABLE} is not set",
)


@pytest.mark.api
@requires_key
@pytest.mark.parametrize("model", ct.CLAUDE_MODELS)
def test_api_counts_tokens(model):
    counter = ct.ClaudeTokenCounter(model)
    short = counter.count("Hello, world!")
    assert isinstance(short, int) and short > 0
    assert counter.count("Hello, world! " * 20) > short


@pytest.mark.api
@requires_key
@pytest.mark.parametrize("model", ct.CLAUDE_MODELS)
def test_api_counting_is_deterministic(model):
    counter = ct.ClaudeTokenCounter(model)
    text = "Искусственный интеллект меняет то, как люди работают с текстом."
    assert counter.count(text) == counter.count(text)


@pytest.mark.api
@requires_key
def test_api_russian_costs_more_than_english():
    counter = ct.ClaudeTokenCounter("claude-opus-5-5")
    assert counter.count(exp.PHRASES["ru"]) > counter.count(exp.PHRASES["en"])


@pytest.mark.api
@requires_key
def test_api_unknown_model_raises_not_found():
    with pytest.raises(anthropic.NotFoundError):
        ct.ClaudeTokenCounter("claude-does-not-exist").count("hello")


@pytest.mark.api
@requires_key
def test_api_invalid_key_is_reported(monkeypatch):
    monkeypatch.setenv(ct.KEY_VARIABLE, "invalid-key-for-test")  # deliberately not key-shaped
    with pytest.raises(anthropic.AuthenticationError):
        ct.ClaudeTokenCounter("claude-haiku-5-5").count("hello")
