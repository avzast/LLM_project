"""Count tokens for Claude models with the official Anthropic SDK (`messages.count_tokens`).

The API key is read ONLY from the ANTHROPIC_API_KEY environment variable (a local `.env`
file is loaded into the environment if python-dotenv is installed; real environment
variables win over it). The key is never written to code, files or output.

Usage:
    python claude_tokens.py "Привет, мир!"              # one text, all models
    python claude_tokens.py --model claude-haiku-5-5 "Hello"
    python claude_tokens.py --corpus                     # the experiment-1 corpus, saved to JSON

The API only returns the number of tokens, not the tokens themselves, and the number is that
of a whole one-message request, so it includes a small constant overhead for the message
framing. Compare texts with each other, not with the tiktoken numbers digit by digit.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

import experiment1_tokenizer as exp

# Current models (ids as published by Anthropic); override with --model.
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
KEY_VARIABLE = "ANTHROPIC_API_KEY"
CLAUDE_RESULTS_FILE = Path(__file__).resolve().parent / "experiment1_claude_results.json"
DOTENV_FILE = Path(__file__).resolve().parent / ".env"

EXIT_OK, EXIT_API_ERROR, EXIT_NO_KEY = 0, 1, 2


class MissingApiKeyError(RuntimeError):
    """ANTHROPIC_API_KEY is not set (or empty)."""


def load_dotenv_if_available(path=DOTENV_FILE):
    """Load `.env` into os.environ without overriding variables that are already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(path, override=False)


def get_api_key(environ=None):
    """Return the key from the environment, or raise MissingApiKeyError with instructions."""
    environ = os.environ if environ is None else environ
    key = (environ.get(KEY_VARIABLE) or "").strip()
    if not key:
        raise MissingApiKeyError(
            f"Не найден ключ API: переменная окружения {KEY_VARIABLE} не задана.\n"
            f"  PowerShell:  $env:{KEY_VARIABLE} = '<ваш ключ>'\n"
            f"  или скопируйте .env.example в .env и впишите ключ туда (.env не попадает в git).\n"
            "Ключ можно создать в консоли Anthropic (Settings -> API keys)."
        )
    return key


def make_client(environ=None):
    """Anthropic client that uses the key from the environment and no other credentials."""
    return anthropic.Anthropic(api_key=get_api_key(environ))


class ClaudeTokenCounter:
    """Number of input tokens of a text for one Claude model, via the count_tokens endpoint."""

    def __init__(self, model, client=None):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:  # created on first use, so constructing a counter never fails
            self._client = make_client()
        return self._client

    def count(self, text):
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return 0  # the API rejects empty / whitespace-only text blocks, so don't ask
        response = self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    def __repr__(self):
        return f"ClaudeTokenCounter(model={self.model!r})"


class _CountOnly:
    """Adapter so that experiment1_tokenizer.run_corpus (which only needs len(ids)) can be reused."""

    def __init__(self, counter):
        self.name = counter.model
        self._counter = counter

    def encode_ids(self, text):
        return range(self._counter.count(text))


def count_corpus(counter, corpus=exp.CORPUS):
    """Same statistics as in experiment 1 (per text, mean/std, pooled), for one Claude model."""
    return exp.run_corpus(_CountOnly(counter), corpus)


def describe_api_error(error):
    """One understandable line for the errors a user can fix; the model/key are named, never shown."""
    if isinstance(error, anthropic.AuthenticationError):
        return f"Ключ из {KEY_VARIABLE} отклонён (неверный или отозванный). Проверьте ключ."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "У ключа нет доступа к этой операции или модели."
    if isinstance(error, anthropic.NotFoundError):
        return "Модель не найдена: проверьте идентификатор модели."
    if isinstance(error, anthropic.RateLimitError):
        return "Превышен лимит запросов. Подождите и повторите."
    if isinstance(error, anthropic.APIConnectionError):
        return "Нет соединения с API Anthropic. Проверьте сеть и прокси."
    if isinstance(error, anthropic.APIStatusError):
        return f"Ошибка API (HTTP {error.status_code}): {error.message}"
    return f"Ошибка: {error}"


def build_parser():
    p = argparse.ArgumentParser(description="Count tokens for Claude models (Anthropic count_tokens API).")
    p.add_argument("text", nargs="?", help="text to count (omit with --corpus)")
    p.add_argument("--model", action="append", help=f"model id, repeatable (default: {', '.join(CLAUDE_MODELS)})")
    p.add_argument("--corpus", action="store_true",
                   help=f"count the experiment-1 corpus and save {CLAUDE_RESULTS_FILE.name}")
    return p


def main(argv=None, environ=None, load_dotenv=True, client=None):
    """CLI entry point; returns an exit code and never lets a missing key or API error traceback out."""
    args = build_parser().parse_args(argv)
    if args.corpus == (args.text is not None):
        print("Укажите либо текст, либо --corpus.", file=sys.stderr)
        return EXIT_API_ERROR
    if load_dotenv and environ is None:
        load_dotenv_if_available()
    try:
        client = client or make_client(environ)
    except MissingApiKeyError as e:
        print(e, file=sys.stderr)
        return EXIT_NO_KEY

    models = args.model or CLAUDE_MODELS
    try:
        if args.corpus:
            results = {m: count_corpus(ClaudeTokenCounter(m, client)) for m in models}
            CLAUDE_RESULTS_FILE.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            for model, res in results.items():
                print(f"=== {model} ===")
                for lang, s in res["languages"].items():
                    print(f"  {lang:7} tokens/word {s['tokens_per_word_mean']:.2f} ± "
                          f"{s['tokens_per_word_std']:.2f}  total {s['total_tokens']}")
                for key, r in res["ratios"].items():
                    print(f"  {key:16} pooled {r['pooled']:.2f}")
            print(f"Сохранено: {CLAUDE_RESULTS_FILE}")
        else:
            for model in models:
                print(f"{model}: {ClaudeTokenCounter(model, client).count(args.text)} токенов")
    except anthropic.APIError as e:
        print(describe_api_error(e), file=sys.stderr)
        return EXIT_API_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")  # Russian messages must survive a cp1252 console
    sys.exit(main())
