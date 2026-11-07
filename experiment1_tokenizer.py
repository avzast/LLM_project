"""Experiment 1: tokenize the same phrase in Russian and English."""
import json
import sys

import truststore

truststore.inject_into_ssl()  # use the OS certificate store for downloading vocab files

import tiktoken

sys.stdout.reconfigure(encoding="utf-8")

PHRASES = {
    "ru": "Искусственный интеллект меняет то, как люди работают с текстом.",
    "en": "Artificial intelligence is changing how people work with text.",
}
ENCODINGS = ["cl100k_base", "o200k_base"]  # GPT-4 / GPT-4o tokenizers


def show(text, enc):
    ids = enc.encode(text)
    # decode each token separately; bytes that are not valid UTF-8 on their own
    # (a Cyrillic letter can be split across tokens) are shown as hex
    pieces = []
    for i in ids:
        b = enc.decode_single_token_bytes(i)
        try:
            pieces.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            pieces.append("<0x" + b.hex() + ">")
    return ids, pieces


results = {}
for enc_name in ENCODINGS:
    enc = tiktoken.get_encoding(enc_name)
    results[enc_name] = {}
    print(f"=== {enc_name} ===")
    for lang, text in PHRASES.items():
        ids, pieces = show(text, enc)
        n_bytes = len(text.encode("utf-8"))
        results[enc_name][lang] = {
            "text": text,
            "chars": len(text),
            "words": len(text.split()),
            "tokens": len(ids),
            "chars_per_token": round(len(text) / len(ids), 2),
            "tokens_per_word": round(len(ids) / len(text.split()), 2),
            "pieces": pieces,
            "ids": ids,
        }
        r = results[enc_name][lang]
        print(f"[{lang}] {text}")
        print(f"  chars={r['chars']} words={r['words']} tokens={r['tokens']} "
              f"chars/token={r['chars_per_token']} tokens/word={r['tokens_per_word']}")
        print("  " + " | ".join(repr(p) for p in pieces))
    ratio = results[enc_name]["ru"]["tokens"] / results[enc_name]["en"]["tokens"]
    results[enc_name]["ru_en_ratio"] = round(ratio, 2)
    print(f"  RU/EN token ratio: {ratio:.2f}\n")

with open("experiment1_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
