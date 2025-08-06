#!/usr/bin/env python3
import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
TWEET_COLUMN = "Text"  # change if the tweet text lives in another column
ENCODING = "utf-8"  # use your csv / txt encoding
CSV_GLOB = "*.csv"  # pattern that matches your 43 emoji csv files


# ────────────────────────────────────────────────────────────────────────────────

def load_wordlist(path: pathlib.Path) -> set[str]:
    """Read a newline-separated wordlist into a lowercase set."""
    return {w.strip().lower() for w in path.read_text(ENCODING).splitlines() if w.strip()}

def robust_read_csv(path: pathlib.Path) -> pd.DataFrame:
    """Try several encodings and fall back to python engine if needed."""
    for enc in ("utf-8", "utf-8-sig", "windows-1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError):
            pass
    # last resort – slow but forgiving
    return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")


here = pathlib.Path(__file__).resolve().parent  # folder containing the script
pos_words = load_wordlist(here / "positive-words.txt")
neg_words = load_wordlist(here / "negative-words.txt")

ratios: list[float | None] = []
labels: list[str] = []

word_re = re.compile(r"[a-zA-Z']+")

here = pathlib.Path("../archive").resolve()
for csv_path in sorted(here.glob(CSV_GLOB)):
    print(f"Loading {csv_path.name}")
    df = robust_read_csv(csv_path)
    text = " ".join(df[TWEET_COLUMN].astype(str))

    tokens = word_re.findall(text.lower())
    pos_count = sum(tok in pos_words for tok in tokens)
    neg_count = sum(tok in neg_words for tok in tokens)

    ratio = pos_count / neg_count if neg_count else None
    ratios.append(ratio)
    labels.append(csv_path.stem)  # e.g. "thumbs_up"

# ── PLOT ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))

ax.barh(labels, [r if r is not None else 0 for r in ratios])
ax.set_xlabel("positive-word count  ÷  negative-word count")
ax.set_title("Pos/Neg Word Ratio per Emoji (all tweets combined)")

# Mark bars where neg_count was zero
for y, ratio in enumerate(ratios):
    if ratio is None:
        ax.text(0.02, y, "no neg words", va="center", ha="left", fontsize=8, color="crimson",
                transform=ax.get_yaxis_transform())

ax.margins(y=0.01)
plt.tight_layout()
plt.show()
