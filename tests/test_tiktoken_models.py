"""Which tiktoken encodings / OpenAI models the project supports.

Everything here is checked against the installed tiktoken, so a model or encoding
that the library does not know is never assumed to exist.
"""
import pytest
import tiktoken

import experiment1_tokenizer as exp

# model name -> encoding, as of tiktoken 0.14 (exact names and prefixes both occur)
MODEL_TO_ENCODING = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-0613": "cl100k_base",  # by prefix
    "text-embedding-3-small": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",  # by prefix
    "gpt-4.1": "o200k_base",
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",  # by prefix
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-oss-20b": "o200k_harmony",
    "gpt-oss-120b": "o200k_harmony",
}


def count_tokens(tok, text):
    return len(tok.encode_ids(text))


def all_corpus_texts():
    return [t for texts in exp.CORPUS.values() for t in texts]


# --- the configured encodings really exist -----------------------------------

@pytest.mark.parametrize("name", exp.ENCODINGS)
def test_configured_encodings_exist_in_installed_tiktoken(name):
    assert name in tiktoken.list_encoding_names()


def test_current_openai_encodings_are_available():
    available = tiktoken.list_encoding_names()
    for name in ("cl100k_base", "o200k_base", "o200k_harmony"):
        assert name in available


def test_no_unknown_encoding_is_loaded():
    with pytest.raises(ValueError):
        exp.TiktokenTokenizer("o300k_base")  # does not exist in tiktoken


# --- o200k_harmony (gpt-oss) ------------------------------------------------

@pytest.fixture(scope="module")
def harmony():
    return exp.load_tokenizer("o200k_harmony")


@pytest.fixture(scope="module")
def o200k():
    return exp.load_tokenizer("o200k_base")


def test_harmony_loads_through_the_common_interface(harmony):
    assert harmony.name == "o200k_harmony"
    assert harmony.encode_ids("Hello, world!") == harmony.encode_ids("Hello, world!")
    assert len(harmony.encode("Hello, world!")) == count_tokens(harmony, "Hello, world!")


def test_harmony_vocabulary_extends_o200k(harmony, o200k):
    assert harmony._enc.n_vocab > o200k._enc.n_vocab
    assert len(harmony._enc.special_tokens_set) > len(o200k._enc.special_tokens_set)


@pytest.mark.parametrize("text", all_corpus_texts())
def test_harmony_counts_plain_text_like_o200k_base(harmony, o200k, text):
    # only special tokens differ, so ordinary text must tokenize identically
    assert harmony.encode_ids(text) == o200k.encode_ids(text)


def test_harmony_special_token_strings_count_as_plain_text(harmony):
    # "<|start|>" is a special token of this encoding, but in user text it is just characters
    assert count_tokens(harmony, "<|start|>") > 1


# --- model name -> tokenizer --------------------------------------------------

@pytest.mark.parametrize("model,encoding", MODEL_TO_ENCODING.items())
def test_tokenizer_for_model_picks_the_right_encoding(model, encoding):
    assert exp.tokenizer_for_model(model).name == encoding


@pytest.mark.parametrize("model", MODEL_TO_ENCODING)
def test_tokenizer_for_model_agrees_with_tiktoken(model):
    text = "Привет, world! Hello, мир."
    ours = exp.tokenizer_for_model(model).encode_ids(text)
    theirs = tiktoken.encoding_for_model(model).encode(text, disallowed_special=())
    assert ours == theirs


@pytest.mark.parametrize("model", ["claude-3-opus", "llama-3", "not-a-model", ""])
def test_tokenizer_for_unknown_model_raises_key_error(model):
    with pytest.raises(KeyError):
        exp.tokenizer_for_model(model)


@pytest.mark.parametrize("bad", [None, 123])
def test_tokenizer_for_non_string_model_raises(bad):
    with pytest.raises((TypeError, KeyError, AttributeError)):
        exp.tokenizer_for_model(bad)


# --- newer encodings should not be worse on Russian / Kazakh ------------------

@pytest.mark.parametrize("lang", ["ru", "kk"])
def test_o200k_needs_fewer_tokens_than_cl100k_on_cyrillic(lang):
    old = exp.load_tokenizer("cl100k_base")
    new = exp.load_tokenizer("o200k_base")
    for text in exp.CORPUS[lang]:
        assert count_tokens(new, text) <= count_tokens(old, text)


def test_model_mapping_in_this_file_matches_tiktoken_tables():
    # guards the table above against going stale after a tiktoken upgrade
    for model, encoding in MODEL_TO_ENCODING.items():
        assert tiktoken.encoding_name_for_model(model) == encoding
