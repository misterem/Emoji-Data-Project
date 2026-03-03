

# ------------------------------------------------------
# Minimal preprocessing:
# - Robust CSV reading (handles encoding/tokenizing issues)
# - Emoji extraction from text
# - Build co-occurrence edge list
# - Save edge list to outputs/tables
# ------------------------------------------------------

import os
import re
import io
from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd

# ------------------------------------------------------
# Robust CSV Reader
# ------------------------------------------------------
def robust_read_csv(path_str: str) -> pd.DataFrame:
    """Try multiple strategies to read problematic CSV files."""
    try:
        return pd.read_csv(path_str, encoding="utf-8", low_memory=False)
    except Exception:
        pass

    try:
        return pd.read_csv(path_str, encoding="utf-8",
                           engine="python", on_bad_lines="skip")
    except Exception:
        pass

    try:
        return pd.read_csv(path_str, encoding="utf-8",
                           engine="python", on_bad_lines="skip",
                           sep=",", quotechar='"', escapechar="\\")
    except Exception:
        pass

    try:
        with open(path_str, "r", encoding="utf-8", errors="ignore", newline="") as f:
            raw = f.read()
        raw = raw.replace("\x00", "")
        buf = io.StringIO(raw)
        return pd.read_csv(buf, engine="python", on_bad_lines="skip",
                           sep=",", quotechar='"', escapechar="\\")
    except Exception as e:
        print(f"[WARN] Failed to parse CSV even after cleanup: {path_str} -> {e}")
        return pd.DataFrame()

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
# Process all CSV files
# ------------------------------------------------------
folder = "../archive"   # relative path from src/
files = sorted([str(p) for p in Path(folder).rglob("*.csv")])
print(f"[INFO] Found {len(files)} csv files")

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
# Save Edge List
# ------------------------------------------------------
if not pair_counts:
    print("[ERROR] No co-occurrence pairs found. Nothing to save.")
    raise SystemExit(0)

os.makedirs("../outputs/tables", exist_ok=True)
edges_path = "../outputs/tables/edges_cooccurrence.csv"

edges = [{"node_u": e1, "node_v": e2, "weight": int(c)}
         for (e1, e2), c in pair_counts.items() if c > 0]
edges_df = pd.DataFrame(edges)
edges_df.to_csv(edges_path, index=False, encoding="utf-8")
print(f"[OK] Saved edges -> {edges_path}")
