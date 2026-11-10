"""Experiment 1: tokenize the same phrase in Russian, Kazakh and English."""
import json
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import truststore
import tiktoken

# --- data ------------------------------------------------------------------

# The four phrases of the original experiment (detailed token-by-token breakdown).
PHRASES = {
    "ru": "Искусственный интеллект меняет то, как люди работают с текстом.",
    "ru_lat": "Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut s tekstom.",  # RU in Latin letters
    "kk": "Жасанды интеллект адамдардың мәтінмен жұмыс істеу тәсілін өзгертіп жатыр.",  # Kazakh
    "en": "Artificial intelligence is changing how people work with text.",
}

# Parallel corpus: text number i means the same in every language.
# The Kazakh texts are hand-written translations (not checked by a native speaker),
# so they may contain mistakes. Text 1 is the phrase from PHRASES; text 6 is a paragraph.
CORPUS_SOURCE = {
    "ru": [
        PHRASES["ru"],
        "Сегодня хорошая погода, и мы пошли гулять в парк.",
        "Столица Казахстана — город Астана.",
        "Я люблю читать книги и разговаривать с друзьями.",
        "Язык — душа народа, и беречь его — долг каждого из нас.",
        "Современные большие языковые модели делят текст не на слова, а на небольшие части, "
        "которые называются токенами. Слова русского и казахского языков часто длинные, "
        "поэтому на одно слово уходит несколько токенов. Это влияет на стоимость запроса "
        "и на скорость заполнения контекстного окна.",
    ],
    "kk": [
        PHRASES["kk"],
        "Бүгін ауа райы жақсы, біз серуендеуге шықтық.",
        "Қазақстанның астанасы — Астана қаласы.",
        "Мен кітап оқуды және достарыммен әңгімелесуді жақсы көремін.",
        "Тіл — халықтың жаны, ал оны сақтау әрқайсымыздың міндетіміз.",
        "Қазіргі заманда үлкен тілдік модельдер мәтінді сөздерге емес, токендер деп аталатын "
        "кішкентай бөліктерге бөледі. Қазақ тіліндегі сөздер көбіне ұзын болады, сондықтан "
        "бір сөзге бірнеше токен жұмсалады. Бұл сұраныстың құнына және контекстік терезенің "
        "толу жылдамдығына әсер етеді.",
    ],
    "en": [
        PHRASES["en"],
        "The weather is nice today, so we went for a walk in the park.",
        "The capital of Kazakhstan is the city of Astana.",
        "I love reading books and talking with friends.",
        "Language is the soul of a nation, and preserving it is the duty of each of us.",
        "Modern large language models split text not into words but into small pieces called "
        "tokens. Words in Russian and Kazakh are often long, so one word takes several tokens. "
        "This affects the cost of a request and how fast the context window fills up.",
    ],
}

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def transliterate(text):
    """Russian -> Latin letters (the scheme used for PHRASES['ru_lat'])."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in _TRANSLIT:
            t = _TRANSLIT[low]
            out.append(t.capitalize() if ch != low else t)
        else:
            out.append(ch)
    return "".join(out)


CORPUS = {
    "ru": CORPUS_SOURCE["ru"],
    "ru_lat": [transliterate(t) for t in CORPUS_SOURCE["ru"]],
    "kk": CORPUS_SOURCE["kk"],
    "en": CORPUS_SOURCE["en"],
}

# --- tokenizers config (add new ones here) -----------------------------------

ENCODINGS = ["cl100k_base", "o200k_base"]  # GPT-4 / GPT-4o tokenizers (tiktoken)
# Hugging Face tokenizers that download without an access token: results name -> repo id
HF_BYTELEVEL = {"qwen2.5": "Qwen/Qwen2.5-0.5B"}  # byte-level BPE, same family as tiktoken
HF_SENTENCEPIECE = {  # SentencePiece-style vocabularies ("▁" marks a word start)
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
TOKENIZER_NAMES = ENCODINGS + list(HF_BYTELEVEL) + list(HF_SENTENCEPIECE)

RESULTS_FILE = Path(__file__).resolve().parent / "experiment1_results.json"  # next to the script

# (key in results, numerator language, denominator language)
RATIOS = [
    ("ru_en_ratio", "ru", "en"),
    ("ru_lat_en_ratio", "ru_lat", "en"),
    ("ru_lat_ru_ratio", "ru_lat", "ru"),
    ("kk_en_ratio", "kk", "en"),
    ("kk_ru_ratio", "kk", "ru"),
]


def inject_system_certs():
    """Use the OS certificate store for downloading vocab files."""
    truststore.inject_into_ssl()


# --- tokenizers ------------------------------------------------------------

def _piece_from_bytes(b):
    # bytes that are not valid UTF-8 on their own (a Cyrillic letter can be
    # split across tokens) are shown as hex
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "<0x" + b.hex() + ">"


class Tokenizer:
    """Common interface: a name, ids for a text, and the token strings."""

    name: str

    def encode_ids(self, text):
        raise NotImplementedError

    def encode(self, text):
        """Return the text split into token strings."""
        raise NotImplementedError


class TiktokenTokenizer(Tokenizer):
    def __init__(self, name):
        inject_system_certs()
        self.name = name
        self._enc = tiktoken.get_encoding(name)

    def encode_ids(self, text):
        return self._enc.encode(text)

    def encode(self, text):
        return [_piece_from_bytes(self._enc.decode_single_token_bytes(i))
                for i in self.encode_ids(text)]


def _bytes_to_unicode():
    """GPT-2 byte -> printable char table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_UNICODE_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}
_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class HFTokenizer(Tokenizer):
    """Base for Hugging Face `tokenizers` models loaded from a repo's tokenizer.json."""

    def __init__(self, name, repo):
        inject_system_certs()
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer as _HFTokenizer

        self.name = name
        self.repo = repo
        self._tok = _HFTokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))

    def _tokens(self, text):
        return self._tok.encode(text, add_special_tokens=False).tokens

    def encode_ids(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids


class HFByteLevelTokenizer(HFTokenizer):
    """Byte-level BPE vocabulary (Qwen, GPT-2 style)."""

    def encode(self, text):
        return [_piece_from_bytes(bytes(_UNICODE_TO_BYTE[c] for c in t))
                for t in self._tokens(text)]


class HFSentencePieceTokenizer(HFTokenizer):
    """SentencePiece-style vocabulary: "▁" is a space, "<0xNN>" is a raw byte (byte fallback)."""

    def encode(self, text):
        pieces = []
        for t in self._tokens(text):
            m = _BYTE_TOKEN.match(t)
            pieces.append("<0x" + m.group(1).lower() + ">" if m else t.replace("▁", " "))
        # the model adds a dummy space before the first word; the text has none
        if pieces and pieces[0].startswith(" "):
            pieces[0] = pieces[0][1:]
        return pieces


def load_tokenizer(name):
    if name in HF_BYTELEVEL:
        return HFByteLevelTokenizer(name, HF_BYTELEVEL[name])
    if name in HF_SENTENCEPIECE:
        return HFSentencePieceTokenizer(name, HF_SENTENCEPIECE[name])
    return TiktokenTokenizer(name)


def load_tokenizers():
    return [load_tokenizer(n) for n in TOKENIZER_NAMES]


# --- metrics ---------------------------------------------------------------

def compute_metrics(text, ids, pieces):
    words = len(text.split())
    return {
        "text": text,
        "chars": len(text),
        "words": words,
        "tokens": len(ids),
        "chars_per_token": round(len(text) / len(ids), 2),
        "tokens_per_word": round(len(ids) / words, 2),
        "pieces": pieces,
        "ids": ids,
    }


def compute_ratios(results):
    return {key: round(results[a]["tokens"] / results[b]["tokens"], 2) for key, a, b in RATIOS}


def _mean_std(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(statistics.mean(values), 2), round(std, 2)


def run_corpus(tok, corpus=CORPUS):
    """Token counts for every text of every language, plus mean/std over the texts.

    Besides the mean of per-text ratios (with std), `pooled_*` values divide the total
    number of tokens by the total number of words (or by the other language's tokens),
    so that long texts weigh more than short ones.
    """
    if len({len(texts) for texts in corpus.values()}) > 1:
        raise ValueError("corpus must be parallel: every language needs the same number of texts")
    per_lang = {}
    for lang, texts in corpus.items():
        items = []
        for text in texts:
            n_tokens = len(tok.encode_ids(text))
            items.append({
                "chars": len(text),
                "words": len(text.split()),
                "tokens": n_tokens,
                "chars_per_token": round(len(text) / n_tokens, 2),
                "tokens_per_word": round(n_tokens / len(text.split()), 2),
            })
        # mean/std from exact values, not from the rounded ones stored above
        tpw_mean, tpw_std = _mean_std([i["tokens"] / i["words"] for i in items])
        cpt_mean, cpt_std = _mean_std([i["chars"] / i["tokens"] for i in items])
        total_tokens = sum(i["tokens"] for i in items)
        total_words = sum(i["words"] for i in items)
        per_lang[lang] = {
            "n_texts": len(items),
            "total_tokens": total_tokens,
            "total_words": total_words,
            "tokens_per_word_mean": tpw_mean,
            "tokens_per_word_std": tpw_std,
            "pooled_tokens_per_word": round(total_tokens / total_words, 2),
            "chars_per_token_mean": cpt_mean,
            "chars_per_token_std": cpt_std,
            "texts": items,
        }
    ratios = {}
    for key, a, b in RATIOS:
        if a in per_lang and b in per_lang:
            each = [x["tokens"] / y["tokens"] for x, y in zip(per_lang[a]["texts"], per_lang[b]["texts"])]
            mean, std = _mean_std(each)
            ratios[key] = {
                "mean": mean,
                "std": std,
                "pooled": round(per_lang[a]["total_tokens"] / per_lang[b]["total_tokens"], 2),
            }
    return {"languages": per_lang, "ratios": ratios}


def run_tokenizer(tok, phrases=PHRASES, corpus=CORPUS):
    results = {}
    for lang, text in phrases.items():
        ids = tok.encode_ids(text)
        pieces = tok.encode(text)
        results[lang] = compute_metrics(text, ids, pieces)
    results.update(compute_ratios(results))
    if corpus:
        results["corpus"] = run_corpus(tok, corpus)
    return results


# --- output ----------------------------------------------------------------

def print_report(name, results):
    print(f"=== {name} ===")
    for lang in PHRASES:
        r = results[lang]
        print(f"[{lang}] {r['text']}")
        print(f"  chars={r['chars']} words={r['words']} tokens={r['tokens']} "
              f"chars/token={r['chars_per_token']} tokens/word={r['tokens_per_word']}")
        print("  " + " | ".join(repr(p) for p in r["pieces"]))
    t = {k: results[k]["tokens"] for k in PHRASES}
    print(f"  RU/EN: {t['ru'] / t['en']:.2f}  RU_LAT/EN: {t['ru_lat'] / t['en']:.2f}  "
          f"RU_LAT/RU: {t['ru_lat'] / t['ru']:.2f}  KK/EN: {t['kk'] / t['en']:.2f}  "
          f"KK/RU: {t['kk'] / t['ru']:.2f}")
    c = results.get("corpus")
    if c:
        print("  corpus (tokens/word, mean ± std over texts; pooled):")
        for lang, s in c["languages"].items():
            print(f"    {lang:7} {s['tokens_per_word_mean']:.2f} ± {s['tokens_per_word_std']:.2f}  "
                  f"pooled {s['pooled_tokens_per_word']:.2f}")
        for key, r in c["ratios"].items():
            print(f"    {key:16} {r['mean']:.2f} ± {r['std']:.2f}  pooled {r['pooled']:.2f}")
    print()


def save_results(results, path=RESULTS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = {}
    for tok in load_tokenizers():
        results[tok.name] = run_tokenizer(tok)
        print_report(tok.name, results[tok.name])
    save_results(results)


if __name__ == "__main__":
    main()
