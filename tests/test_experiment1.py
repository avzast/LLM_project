import json
from pathlib import Path

import pytest

import experiment1_tokenizer as exp

RESULTS_PATH = exp.RESULTS_FILE


class FakeTokenizer(exp.Tokenizer):
    """Splits on whitespace: one token per word, no network or vocab files."""

    name = "fake"

    def encode(self, text):
        return text.split()

    def encode_ids(self, text):
        return list(range(len(text.split())))


def pieces_to_text(pieces):
    """Rebuild the text; '<0x..>' pieces are raw bytes of a split UTF-8 character."""
    raw = b""
    for p in pieces:
        if p.startswith("<0x") and p.endswith(">"):
            raw += bytes.fromhex(p[3:-1])
        else:
            raw += p.encode("utf-8")
    return raw.decode("utf-8")


# --- metrics on synthetic examples ---------------------------------------

def test_compute_metrics_simple():
    text = "aa bb cc dd"
    m = exp.compute_metrics(text, [1, 2, 3, 4, 5, 6, 7, 8], list("abcdefgh"))
    assert m["chars"] == 11
    assert m["words"] == 4
    assert m["tokens"] == 8
    assert m["chars_per_token"] == round(11 / 8, 2)  # 1.38
    assert m["tokens_per_word"] == 2.0


def test_compute_metrics_rounding_and_cyrillic():
    text = "привет мир"  # 10 chars, 2 words
    m = exp.compute_metrics(text, [1, 2, 3], ["при", "вет", " мир"])
    assert m["chars"] == 10
    assert m["words"] == 2
    assert m["chars_per_token"] == 3.33
    assert m["tokens_per_word"] == 1.5


def test_compute_ratios():
    results = {
        "ru": {"tokens": 30},
        "ru_lat": {"tokens": 24},
        "kk": {"tokens": 60},
        "en": {"tokens": 10},
    }
    r = exp.compute_ratios(results)
    assert r == {
        "ru_en_ratio": 3.0,
        "ru_lat_en_ratio": 2.4,
        "ru_lat_ru_ratio": 0.8,
        "kk_en_ratio": 6.0,
        "kk_ru_ratio": 2.0,
    }


def test_run_tokenizer_with_fake_tokenizer():
    phrases = {"ru": "раз два три", "ru_lat": "raz dva tri", "kk": "бір екі үш", "en": "one two three"}
    res = exp.run_tokenizer(FakeTokenizer(), phrases, corpus=None)
    for lang in phrases:
        assert res[lang]["tokens"] == 3
        assert res[lang]["tokens_per_word"] == 1.0
    assert res["ru_en_ratio"] == 1.0
    assert set(res) == set(phrases) | {key for key, _, _ in exp.RATIOS}


def test_run_corpus_mean_std_and_pooled():
    corpus = {"ru": ["a b", "a b c d"], "en": ["a", "a b"]}  # FakeTokenizer: tokens == words
    res = exp.run_corpus(FakeTokenizer(), corpus)
    ru = res["languages"]["ru"]
    assert ru["total_tokens"] == 6 and ru["total_words"] == 6
    assert ru["tokens_per_word_mean"] == 1.0 and ru["tokens_per_word_std"] == 0.0
    assert ru["pooled_tokens_per_word"] == 1.0
    assert [t["tokens"] for t in ru["texts"]] == [2, 4]
    r = res["ratios"]["ru_en_ratio"]  # per text: 2/1 and 4/2
    assert r["mean"] == 2.0 and r["std"] == 0.0 and r["pooled"] == 2.0
    assert "kk_en_ratio" not in res["ratios"]  # languages missing from the corpus are skipped


def test_run_corpus_rejects_non_parallel_corpus():
    with pytest.raises(ValueError):
        exp.run_corpus(FakeTokenizer(), {"ru": ["a b"], "en": ["a", "b"]})


def test_transliterate():
    assert exp.transliterate(exp.PHRASES["ru"]) == exp.PHRASES["ru_lat"]
    assert exp.transliterate("Щука, ёж, Юля") == "Shchuka, yozh, Yulya"
    assert exp.transliterate("Hello, мир!") == "Hello, mir!"  # other characters stay as they are


def test_corpus_is_parallel():
    sizes = {lang: len(texts) for lang, texts in exp.CORPUS.items()}
    assert set(sizes) == {"ru", "ru_lat", "kk", "en"}
    assert len(set(sizes.values())) == 1 and sizes["ru"] >= 5
    for lang in exp.PHRASES:
        assert exp.CORPUS[lang][0] == exp.PHRASES[lang]
    assert all(t.strip() for texts in exp.CORPUS.values() for t in texts)


def test_pieces_to_text_handles_split_bytes():
    # 'Ж' = D0 96, split into two single-byte tokens
    assert pieces_to_text(["<0xd0>", "<0x96>", "ас"]) == "Жас"


# --- real tokenizers from the config (vocab files must be cached or downloadable) --------

@pytest.fixture(scope="module", params=exp.TOKENIZER_NAMES)
def tok(request):
    return exp.load_tokenizer(request.param)


CORPUS_TEXTS = [(lang, i) for lang, texts in exp.CORPUS.items() for i in range(len(texts))]


@pytest.mark.parametrize("lang,i", CORPUS_TEXTS)
def test_pieces_join_to_original_text(tok, lang, i):
    text = exp.CORPUS[lang][i]
    assert pieces_to_text(tok.encode(text)) == text


@pytest.mark.parametrize("lang", list(exp.PHRASES))
def test_pieces_and_ids_have_same_length(tok, lang):
    text = exp.PHRASES[lang]
    assert len(tok.encode(text)) == len(tok.encode_ids(text))


# --- consistency with the saved experiment1_results.json -----------------

@pytest.fixture(scope="module")
def saved():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", exp.TOKENIZER_NAMES)
def test_results_match_saved_json(saved, name):
    fresh = exp.run_tokenizer(exp.load_tokenizer(name))
    assert fresh == saved[name]


def test_saved_json_covers_all_phrases(saved):
    for enc_name in exp.TOKENIZER_NAMES:
        for lang, text in exp.PHRASES.items():
            assert saved[enc_name][lang]["text"] == text
