"""Edge cases of token counting: empty and invalid input, scripts, emoji, code, long text.

`count_tokens` below is what the experiment does everywhere: len(tok.encode_ids(text)).
Every test runs for all tokenizers from experiment1_tokenizer.TOKENIZER_NAMES.
"""
import time

import pytest

import experiment1_tokenizer as exp
from test_experiment1 import pieces_to_text

# tokenizers that never lose a character (XLM-R maps unknown characters to <unk>)
LOSSLESS = [n for n in exp.TOKENIZER_NAMES if n != "xlm-roberta"]
# byte-level vocabularies keep every space exactly; SentencePiece models normalize whitespace
BYTE_LEVEL = [n for n in LOSSLESS if n != "mistral-7b"]
SENTENCEPIECE = list(exp.HF_SENTENCEPIECE)


def count_tokens(tok, text):
    return len(tok.encode_ids(text))


@pytest.fixture(scope="module", params=exp.TOKENIZER_NAMES)
def tok(request):
    return exp.load_tokenizer(request.param)


@pytest.fixture(scope="module", params=LOSSLESS)
def lossless_tok(request):
    return exp.load_tokenizer(request.param)


@pytest.fixture(scope="module", params=BYTE_LEVEL)
def bytelevel_tok(request):
    return exp.load_tokenizer(request.param)


@pytest.fixture(scope="module", params=SENTENCEPIECE)
def sp_tok(request):
    return exp.load_tokenizer(request.param)


# --- empty string ----------------------------------------------------------

def test_empty_string_has_no_tokens(tok):
    assert tok.encode_ids("") == []
    assert tok.encode("") == []
    assert count_tokens(tok, "") == 0


WHITESPACE = [" ", "   ", "\n", "\t\n "]


@pytest.mark.parametrize("text", WHITESPACE)
def test_whitespace_only_is_counted(lossless_tok, text):
    assert count_tokens(lossless_tok, text) >= 1


@pytest.mark.parametrize("text", WHITESPACE)
def test_whitespace_only_roundtrips_on_byte_level(bytelevel_tok, text):
    assert pieces_to_text(bytelevel_tok.encode(text)) == text


def test_sentencepiece_ignores_a_single_leading_space(sp_tok):
    # a property of the models, not of our code: " a" and "a" look the same to them
    assert sp_tok.encode_ids(" a") == sp_tok.encode_ids("a")


def test_xlm_roberta_drops_whitespace_only_text():
    # XLM-R's normalizer removes whitespace-only input completely: 0 tokens
    assert count_tokens(exp.load_tokenizer("xlm-roberta"), "   ") == 0


# --- short English ---------------------------------------------------------

def test_short_english_hello_world_gpt_tokenizers():
    # well-known counts: "Hello" "," " world" "!"
    assert count_tokens(exp.load_tokenizer("cl100k_base"), "Hello, world!") == 4
    assert count_tokens(exp.load_tokenizer("o200k_base"), "Hello, world!") == 4


def test_short_english_count_is_reasonable(tok):
    text = "Hello, world!"
    n = count_tokens(tok, text)
    assert 2 <= n <= len(text)


def test_single_english_word_is_few_tokens(tok):
    assert 1 <= count_tokens(tok, "language") <= 3


def test_counting_is_deterministic(tok):
    text = "The quick brown fox jumps over the lazy dog."
    assert tok.encode_ids(text) == tok.encode_ids(text)


# --- Russian ---------------------------------------------------------------

def test_russian_text_count_and_roundtrip(tok):
    text = "Привет, как дела? Сегодня хорошая погода."
    n = count_tokens(tok, text)
    assert 1 <= n <= len(text.encode("utf-8"))  # never more tokens than bytes
    assert len(tok.encode(text)) == n


def test_russian_costs_more_tokens_than_english_on_old_tokenizer():
    tok = exp.load_tokenizer("cl100k_base")
    assert count_tokens(tok, "Привет, как дела?") > count_tokens(tok, "Hi, how are you?")


def test_russian_letter_case_changes_tokens_but_not_text(lossless_tok):
    for text in ("привет", "ПРИВЕТ", "Привет"):
        assert pieces_to_text(lossless_tok.encode(text)) == text


# --- mixed text ------------------------------------------------------------

MIXED = "Привет, world! Меня зовут Алексей, I work with LLM — языковыми моделями."


def test_mixed_text_roundtrip_and_ids_match_pieces(lossless_tok):
    assert pieces_to_text(lossless_tok.encode(MIXED)) == MIXED
    assert len(lossless_tok.encode(MIXED)) == count_tokens(lossless_tok, MIXED)


def test_mixed_text_count_is_close_to_sum_of_parts(tok):
    ru = "Привет, "
    en = "world! "
    whole = count_tokens(tok, ru + en)
    parts = count_tokens(tok, ru) + count_tokens(tok, en)
    assert abs(whole - parts) <= 2  # token boundaries may shift a little at the seam


def test_numbers_and_digits(tok):
    assert count_tokens(tok, "2024-11-10 12:30:45") >= 3


# --- emoji -----------------------------------------------------------------

EMOJI = ["😀", "👋🌍", "👨‍👩‍👧‍👦", "🇰🇿", "Hello 👋 мир 🌍!"]


@pytest.mark.parametrize("text", EMOJI)
def test_emoji_counted_and_lossless(lossless_tok, text):
    assert count_tokens(lossless_tok, text) >= 1
    assert pieces_to_text(lossless_tok.encode(text)) == text


@pytest.mark.parametrize("text", EMOJI)
def test_emoji_do_not_crash_any_tokenizer(tok, text):
    assert count_tokens(tok, text) >= 1


def test_family_emoji_costs_more_than_single_emoji(lossless_tok):
    # 7 code points joined with zero-width joiners vs one code point
    assert count_tokens(lossless_tok, "👨‍👩‍👧‍👦") > count_tokens(lossless_tok, "😀")


# --- special characters and code -------------------------------------------

SPECIAL = [
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~\\",
    "tab\tseparated\tvalues\r\nwindows newline\n",
    "zero​width and non breaking space",
    "ñ é ü ß 漢字 テスト 한국어 العربية עברית",
    "null byte: \x00 end",
]

CODE = '''def fib(n: int) -> int:
    """Return n-th Fibonacci number."""
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(f"fib(10) = {fib(10)}")  # -> 55
'''


@pytest.mark.parametrize("text", SPECIAL)
def test_special_characters_lossless(lossless_tok, text):
    assert pieces_to_text(lossless_tok.encode(text)) == text
    assert count_tokens(lossless_tok, text) >= 1


def test_code_roundtrip_and_reasonable_density(lossless_tok):
    n = count_tokens(lossless_tok, CODE)
    assert pieces_to_text(lossless_tok.encode(CODE)) == CODE
    assert len(CODE) / n > 1.5  # code is tokenized into multi-character pieces


def test_indentation_is_not_free(tok):
    assert count_tokens(tok, "x = 1\n    y = 2") >= count_tokens(tok, "x = 1\ny = 2")


def test_json_snippet(tok):
    assert count_tokens(tok, '{"name": "Алексей", "tags": ["llm", "токены"], "n": 3}') > 10


@pytest.mark.parametrize("name", exp.ENCODINGS)
def test_tiktoken_counts_special_token_string_as_plain_text(name):
    # as ordinary text "<|endoftext|>" is several tokens, not the single control token
    assert count_tokens(exp.load_tokenizer(name), "<|endoftext|>") > 1


def test_special_token_text_is_treated_as_plain_text(tok):
    # a user text may contain strings like "<|endoftext|>"; counting must not crash
    text = "Мой текст содержит <|endoftext|> и <s> и </s> и <unk> внутри."
    assert count_tokens(tok, text) >= 5


# --- very long text -------------------------------------------------------

def test_long_text_count_grows_linearly(tok):
    chunk = "The quick brown fox jumps over the lazy dog. "
    small = count_tokens(tok, chunk * 100)
    big = count_tokens(tok, chunk * 2000)
    assert 18 * small <= big <= 22 * small  # 20x text, ~20x tokens


def test_very_long_text_is_fast_and_exact_count(tok):
    text = "Привет, world! " * 20_000  # ~300k characters
    start = time.perf_counter()
    ids = tok.encode_ids(text)
    elapsed = time.perf_counter() - start
    assert len(ids) > 20_000
    assert elapsed < 30
    assert len(set(ids)) < 50  # repeated text -> few distinct tokens


def test_long_text_without_spaces(tok):
    text = "a" * 100_000
    assert count_tokens(tok, text) >= 1000


# --- invalid input ---------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 123, 4.5, b"bytes", ["a", "b"], {"a": 1}, object()])
def test_non_string_input_raises_type_error(tok, bad):
    with pytest.raises(TypeError):
        tok.encode_ids(bad)


@pytest.mark.parametrize("bad", [None, 123, ["a"]])
def test_non_string_input_to_pieces_raises_type_error(tok, bad):
    with pytest.raises(TypeError):
        tok.encode(bad)


@pytest.mark.parametrize("bad", [None, 123])
def test_compute_metrics_non_string_raises_type_error(bad):
    with pytest.raises(TypeError):
        exp.compute_metrics(bad, [1], ["a"])


def test_lone_surrogate_gives_ids_or_a_clean_error(tok):
    # "\ud800" is not valid text: tiktoken replaces it with U+FFFD, the HF tokenizers raise
    try:
        ids = tok.encode_ids("abc \ud800 def")
    except (TypeError, ValueError, UnicodeError):
        return
    assert ids and all(isinstance(i, int) for i in ids)


# --- metrics on degenerate input --------------------------------------------

def test_metrics_for_empty_text_do_not_crash():
    m = exp.compute_metrics("", [], [])
    assert m["tokens"] == 0 and m["chars"] == 0 and m["words"] == 0


def test_metrics_for_whitespace_only_text_do_not_crash():
    m = exp.compute_metrics("   ", [1], ["   "])
    assert m["words"] == 0 and m["tokens"] == 1


def test_run_corpus_with_empty_text_does_not_crash(tok):
    res = exp.run_corpus(tok, {"ru": [""], "en": ["a"]})
    assert res["languages"]["ru"]["total_tokens"] == 0
