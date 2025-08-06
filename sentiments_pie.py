import math
import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────
TWEET_COLUMN = "Text"  # column that holds the tweet text
CSV_GLOB = "*.csv"  # pattern that matches your 43 emoji csv files
ENCODINGS_TRY = ("utf-8", "utf-8-sig", "windows-1252")
# ──────────────────────────────────────────────────────────────────────────

here = pathlib.Path(__file__).resolve().parent


def load_wordlist(path: pathlib.Path) -> set[str]:
    return {w.strip().lower() for w in path.read_text("utf-8").splitlines() if w.strip()}


def robust_read_csv(path: pathlib.Path) -> pd.DataFrame:
    """Try a few encodings; fall back to the tolerant Python parser."""
    for enc in ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # last resort – slow but forgiving
    return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")


# word lists
pos_words = load_wordlist(here / "positive-words.txt")
neg_words = load_wordlist(here / "negative-words.txt")
word_re = re.compile(r"[a-zA-Z']+")

# gather counts
emojis, pos_counts, neg_counts = [], [], []
here = pathlib.Path("./archive").resolve()
for csv_path in sorted(here.glob(CSV_GLOB)):
    df = robust_read_csv(csv_path)
    tokens = word_re.findall(" ".join(df[TWEET_COLUMN].astype(str)).lower())
    p, n = sum(t in pos_words for t in tokens), sum(t in neg_words for t in tokens)
    emojis.append(csv_path.stem)  # e.g. "thumbs_up"
    pos_counts.append(p)
    neg_counts.append(n)

# ── PIE-CHART GRID ────────────────────────────────────────────────────────
n = len(emojis)
ncols = 5  # tweak for wider / narrower grids
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3, nrows * 3),
                         subplot_kw=dict(aspect="equal"))

axes = axes.flatten()
for ax, emoji, p, n in zip(axes, emojis, pos_counts, neg_counts):
    total = p + n
    if total == 0:
        ax.text(0.5, 0.5, "no\nwords", ha="center", va="center", fontsize=8)
        ax.set_title(emoji, fontsize=9)
        ax.axis("off")
        continue

    ax.pie([p, n],
           labels=["positive", "negative"],
           autopct=lambda v: f"{v:.1f}%",
           startangle=90,
           textprops={"fontsize": 7})
    ax.set_title(emoji, fontsize=9)

# hide any unused cells
for ax in axes[len(emojis):]:
    ax.axis("off")

fig.suptitle("Positive vs Negative Word Share per Emoji", fontsize=14)
plt.tight_layout()
plt.show()
