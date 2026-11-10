"""Compare token counts: the same texts through every supported tokenizer (OpenAI, others, Claude).

    python compare_tokenizers.py            # writes results.md and results.csv

Claude columns need ANTHROPIC_API_KEY (see claude_tokens.py). Without a key they are left
empty ("—") and the script says so; the other columns are always computed.
"""
import csv
import sys
from pathlib import Path

import anthropic

import claude_tokens as ct
import experiment1_tokenizer as exp

OUT_DIR = Path(__file__).resolve().parent
MD_FILE = OUT_DIR / "results.md"
CSV_FILE = OUT_DIR / "results.csv"
MISSING = "—"

_PARAGRAPH_EN = (
    "Large language models do not read words, they read tokens: short pieces of text such as "
    "whole words, parts of words, single letters or even raw bytes. The same sentence can cost "
    "a different number of tokens in different languages and in different models. "
)

CODE = '''def fib(n: int) -> int:
    """Return the n-th Fibonacci number."""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


if __name__ == "__main__":
    print([fib(i) for i in range(10)])  # -> [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
'''

TEXTS = {
    "English": "Artificial intelligence is changing how people work with text. "
               "Tokenizers split the text into small pieces before the model reads it.",
    "Russian": "Искусственный интеллект меняет то, как люди работают с текстом. "
               "Токенайзеры делят текст на небольшие части, прежде чем модель его прочитает.",
    "Code": CODE,
    "Mixed": "Привет! I'm using the `tiktoken` library, чтобы посчитать tokens в тексте "
             "из 3 языков (RU/EN/KK): Сәлем, әлем! 📊 price = 4.5 USD/1M",
    "Long": _PARAGRAPH_EN * 40,  # ~10 000 characters
}

# column label for every tokenizer (provider in brackets)
LABELS = {
    "cl100k_base": "cl100k_base (OpenAI GPT-4)",
    "o200k_base": "o200k_base (OpenAI GPT-4o)",
    "qwen2.5": "qwen2.5 (Alibaba)",
    "xlm-roberta": "xlm-roberta (Meta)",
    "mistral-7b": "mistral-7b (Mistral AI)",
}


def count_local(texts=TEXTS, names=exp.TOKENIZER_NAMES):
    """{column label: {text label: tokens}} for the tokenizers that run on this machine."""
    counts = {}
    for name in names:
        tok = exp.load_tokenizer(name)
        counts[LABELS.get(name, name)] = {label: len(tok.encode_ids(text)) for label, text in texts.items()}
    return counts


def count_claude(texts=TEXTS, models=ct.CLAUDE_MODELS, client=None):
    """Same for Claude through the API. Returns (counts, notes): unavailable models are left out."""
    counts, notes = {}, []
    try:
        client = client or ct.make_client()
    except ct.MissingApiKeyError:
        return counts, [f"Claude не посчитан: не задана переменная окружения {ct.KEY_VARIABLE} "
                        "(см. README, раздел «Подсчёт токенов Claude»)."]
    for model in models:
        counter = ct.ClaudeTokenCounter(model, client)
        try:
            counts[f"{model} (Anthropic)"] = {label: counter.count(text) for label, text in texts.items()}
        except anthropic.APIError as e:
            notes.append(f"{model} не посчитан: {ct.describe_api_error(e)}")
    return counts, notes


def claude_column_labels(models=ct.CLAUDE_MODELS):
    return [f"{m} (Anthropic)" for m in models]


def build_rows(counts, columns, texts=TEXTS):
    """Rows: [text label, characters, count per column (None when unavailable)]."""
    return [
        [label, len(text)] + [counts.get(col, {}).get(label) for col in columns]
        for label, text in texts.items()
    ]


def render_markdown(columns, rows, notes=()):
    cell = lambda v: MISSING if v is None else (f"{v:,}".replace(",", " ") if isinstance(v, int) else str(v))
    lines = ["# Сравнение токенайзеров", "",
             "Количество токенов для одних и тех же текстов (меньше = дешевле). "
             "Сгенерировано `compare_tokenizers.py`.", "",
             "| Текст | Символов | " + " | ".join(columns) + " |",
             "|---|---:|" + "---:|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) if i else str(v) for i, v in enumerate(row)) + " |" for row in rows]
    lines += ["",
              "Числа Claude получены через `messages.count_tokens` для запроса из одного сообщения, "
              "поэтому включают небольшую постоянную добавку на служебную разметку; сравнивайте их "
              "между текстами, а не с другими колонками цифра в цифру. «—» означает, что значение "
              "не получено."]
    lines += ["", *[f"- {n}" for n in notes]] if notes else []
    return "\n".join(lines) + "\n"


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so that Excel reads UTF-8
        writer = csv.writer(f)
        writer.writerow(["text", "chars"] + columns)
        writer.writerows([["" if v is None else v for v in row] for row in rows])


def main(client=None, md_file=MD_FILE, csv_file=CSV_FILE):
    ct.load_dotenv_if_available()
    local = count_local()
    claude, notes = count_claude(client=client)
    counts = {**local, **claude}
    columns = list(local) + claude_column_labels()
    rows = build_rows(counts, columns)
    md = render_markdown(columns, rows, notes)
    Path(md_file).write_text(md, encoding="utf-8")
    write_csv(csv_file, columns, rows)
    print(md)
    print(f"Сохранено: {md_file}, {csv_file}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
