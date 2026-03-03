
# ------------------------------------------------------
# - Robust CSV reading (handles encoding/tokenizing issues)
# - Emoji extraction from text
# - Build co-occurrence matrix
# - Save matrix + edge list to outputs/tables
# - Plot heatmap of Top-20 emojis using Apple Color Emoji font (macOS)
# ------------------------------------------------------

import os
import re
import io
from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np

# Visualization
import matplotlib, mplcairo
try:
    matplotlib.use("module://mplcairo.macosx")   # Apple Color Emoji support on macOS
except Exception:
    matplotlib.use("Agg")  # Fallback if mplcairo not available

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

# ------------------------------------------------------
# Robust CSV Reader
# ------------------------------------------------------
def robust_read_csv(path_str: str) -> pd.DataFrame:
    """
    Attempt to read problematic CSV files in multiple stages:
    1) Default C engine
    2) engine='python' with on_bad_lines='skip'
    3) engine='python' with explicit parameters
    4) After cleaning problematic characters
    """
    # Stage 1 – Default C engine
    try:
        return pd.read_csv(path_str, encoding="utf-8", low_memory=False)
    except Exception:
        pass

    # Stage 2 – Python engine, skip bad lines
    try:
        return pd.read_csv(path_str, encoding="utf-8",
                           engine="python", on_bad_lines="skip")
    except Exception:
        pass

    # Stage 3 – Python engine with extra parameters
    try:
        return pd.read_csv(path_str, encoding="utf-8",
                           engine="python", on_bad_lines="skip",
                           sep=",", quotechar='"', escapechar="\\")
    except Exception:
        pass

    # Stage 4 – Clean problematic characters (NULL, etc.)
    try:
        with open(path_str, "r", encoding="utf-8", errors="ignore", newline="") as f:
            raw = f.read()
        raw = raw.replace("\x00", "")  # Remove NULL characters
        buf = io.StringIO(raw)
        return pd.read_csv(buf, engine="python", on_bad_lines="skip",
                           sep=",", quotechar='"', escapechar="\\")
    except Exception as e:
        print(f"[WARN] failed to parse CSV even after cleanup: {path_str} -> {e}")
        return pd.DataFrame()

# ------------------------------------------------------
# Apple Emoji Font Setup (macOS)
# ------------------------------------------------------
APPLE_EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_PROP = None
if Path(APPLE_EMOJI).exists():
    try:
        font_manager.fontManager.addfont(APPLE_EMOJI)
        plt.rcParams["font.family"] = "Apple Color Emoji"
        EMOJI_PROP = FontProperties(fname=APPLE_EMOJI)
    except Exception:
        pass

# ------------------------------------------------------
# Emoji Extraction
# ------------------------------------------------------
try:
    import emoji
    USE_EMOJI_PKG = True
except Exception:
    USE_EMOJI_PKG = False

EMOJI_RE = re.compile(r'[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]')

def extract_emojis(text: str):
    """Extract emojis from text using emoji package if available, else regex."""
    t = str(text or "")
    if USE_EMOJI_PKG:
        try:
            return [c for c in t if c in emoji.EMOJI_DATA]
        except Exception:
            return EMOJI_RE.findall(t)
    return EMOJI_RE.findall(t)

# ------------------------------------------------------
# Read all CSV files from archive/
# ------------------------------------------------------
folder = "../archive"   # relative path from src/
files = sorted([str(p) for p in Path(folder).rglob("*.csv")])
print(f"[INFO] found {len(files)} csv files")

pair_counts = Counter()

for path_str in files:
    df = robust_read_csv(path_str)
    if df.empty or "Text" not in df.columns:
        print(f"[SKIP] no usable data in {path_str}")
        continue

    texts = df["Text"].fillna("").astype(str)
    for text in texts:
        found = set(extract_emojis(text))
        if len(found) > 1:
            for pair in combinations(sorted(found), 2):
                pair_counts[pair] += 1

# ------------------------------------------------------
# Build Co-occurrence Matrix
# ------------------------------------------------------
if not pair_counts:
    print("[ERROR] No co-occurrence pairs found. Nothing to save/plot.")
    raise SystemExit(0)

unique_emojis = sorted({e for pr in pair_counts for e in pr})
matrix = pd.DataFrame(0, index=unique_emojis, columns=unique_emojis, dtype=int)

for (e1, e2), c in pair_counts.items():
    matrix.at[e1, e2] = c
    matrix.at[e2, e1] = c

# ------------------------------------------------------
# Save outputs
# ------------------------------------------------------
os.makedirs("../outputs/tables", exist_ok=True)
mat_path = "../outputs/tables/cooccurrence_matrix.csv"
edges_path = "../outputs/tables/edges_cooccurrence.csv"

matrix.to_csv(mat_path, encoding="utf-8")
print(f"[OK] saved matrix -> {mat_path}")

edges = [{"node_u": e1, "node_v": e2, "weight": int(c)}
         for (e1, e2), c in pair_counts.items() if c > 0]
edges_df = pd.DataFrame(edges)
edges_df.to_csv(edges_path, index=False, encoding="utf-8")
print(f"[OK] saved edges  -> {edges_path}")

# ------------------------------------------------------
# Heatmap Top-20 Emojis
# ------------------------------------------------------
TOP = 20
row_sums = matrix.sum(axis=1)
top_emojis = row_sums.sort_values(ascending=False).head(TOP).index.tolist()

if len(top_emojis) >= 2:
    reduced = matrix.loc[top_emojis, top_emojis]
    logged = np.log1p(reduced)
    normed = (logged - logged.min().min()) / ((logged.max().max() - logged.min().min()) or 1.0)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        normed, cmap="Reds", square=True, linewidths=0.5,
        annot=reduced, fmt="d", annot_kws={"size": 6}
    )

    if EMOJI_PROP is not None:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(EMOJI_PROP)

    plt.title("Top 20 Emoji Co-occurrence Heatmap")
    plt.tight_layout()

    os.makedirs("../outputs/figures", exist_ok=True)
    out_png = "../outputs/figures/heatmap_top20.png"
    plt.savefig(out_png, dpi=300)
    print(f"[OK] saved heatmap -> {out_png}")
else:
    print("[WARN] Not enough emojis to plot a heatmap.")
