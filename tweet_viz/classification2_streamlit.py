import argparse
from collections import Counter
import numpy as np
import pandas as pd
import emoji
from nrclex import NRCLex
import pickle
import os, nltk
from pathlib import Path


def _prepare_nltk_local():
    data_dir = Path("./nltk_data")
    if data_dir.exists():
        os.environ.setdefault("NLTK_DATA", str(data_dir.resolve()))
        if str(data_dir) not in nltk.data.path:
            nltk.data.path.append(str(data_dir))


_prepare_nltk_local()

def save_profiles(profiles, path):
    with open(path, "wb") as f:
        pickle.dump(profiles, f)

def load_profiles(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_profiles(data_dir: Path, n_rows: int = 1000, text_col: str = "text",
                 cache_path: Path | None = None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    # verify required column on a few files
    for p in csvs[:3]:
        try:
            df_head = pd.read_csv(p, nrows=1)
        except Exception as e:
            raise RuntimeError(f"Failed reading {p.name}: {e}")
        if text_col not in df_head.columns:
            raise KeyError(
                f"Required column '{text_col}' missing in {p.name}. "
                f"Columns: {list(df_head.columns)}"
            )

    if cache_path and cache_path.exists():
        return load_profiles(cache_path)

    profiles = build_emoji_profiles(data_dir, n_rows, text_col)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_profiles(profiles, cache_path)
    return profiles

def nrc_emotions(text: str):
    e = NRCLex(text or "")
    raw = dict(e.raw_emotion_scores)
    freq = dict(e.affect_frequencies)
    categories = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive",
                  "negative"]
    raw_full = {k: int(raw.get(k, 0)) for k in categories}
    freq_full = {k: float(freq.get(k, 0.0)) for k in categories}
    return {"raw": raw_full, "freq": freq_full}


# Fixed NRCLex categories (order matters)
EMO_CATS = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

def _nrc_vector(text: str) -> np.ndarray:
    """Return a fixed-length NRCLex emotion vector in EMO_CATS order."""
    if not isinstance(text, str) or not text:
        return np.zeros(len(EMO_CATS), dtype=float)
    s = NRCLex(text).raw_emotion_scores
    return np.array([float(s.get(k, 0.0)) for k in EMO_CATS], dtype=float)

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _extract_emojis(s: str):
    return [c for c in str(s) if c in emoji.EMOJI_DATA]

def _infer_file_emoji(stem: str, texts: pd.Series) -> str | None:
    """Try filename alias then fallback to most common emoji in sample texts."""
    alias = f":{stem.strip().lower().replace(' ', '_').replace('-', '_')}:"
    emj = emoji.emojize(alias, language='alias')
    if emj != alias and emj in emoji.EMOJI_DATA:
        return emj
    counter = Counter()
    for t in texts.dropna().tolist():
        counter.update(_extract_emojis(t))
    return counter.most_common(1)[0][0] if counter else None

def build_emoji_profiles(data_dir: Path, n_rows_per_file: int, text_column: str) -> dict:
    """
    Aggregate NRCLex emotion vectors over the first n rows of each per-emoji CSV.
    Returns: {emoji_char: {"vec": np.ndarray, "count": int, "file": filename}}
    """
    profiles: dict[str, dict] = {}
    csvs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".csv"])
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=n_rows_per_file)
        except Exception:
            continue

        # Pick text column
        if text_column not in df.columns:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            use_col = obj_cols[0] if obj_cols else df.columns[0]
        else:
            use_col = text_column

        # Identify emoji for file
        emj = _infer_file_emoji(p.stem, df[use_col].head(min(len(df), 200)))
        if not emj:
            continue

        # Sum emotion vectors
        total = np.zeros(len(EMO_CATS), dtype=float)
        cnt = 0
        for t in df[use_col].fillna(''):
            total += _nrc_vector(t)
            cnt += 1

        if cnt == 0:
            continue

        profiles[emj] = {"vec": total, "count": cnt, "file": p.name}

    # Normalize vectors to unit length to enable cosine similarity
    for emj, d in profiles.items():
        d["vec"] = _normalize(d["vec"])

    return profiles

def _cosine_details(vec_a: np.ndarray, vec_b: np.ndarray):
    dot = float(np.dot(vec_a, vec_b))
    na = float(np.linalg.norm(vec_a) + 1e-12)
    nb = float(np.linalg.norm(vec_b) + 1e-12)
    cos_sim = dot / (na * nb)
    cos_dist = 1.0 - cos_sim
    return {"dot": dot, "norm_a": na, "norm_b": nb, "cos_sim": cos_sim, "cos_dist": cos_dist}

# modify best_emoji signature and return when requested
def best_emoji(user_text: str, profiles, top_k=1, return_details: bool = False):
    v = _normalize(_nrc_vector(user_text))
    scored = []
    for emj, d in profiles.items():
        details = _cosine_details(v, d["vec"])
        score = details["cos_sim"]
        if return_details:
            scored.append({"emoji": emj, "score": score, **details})
        else:
            scored.append((score, emj))
    scored.sort(key=lambda x: x["score"] if return_details else x[0], reverse=True)
    return scored[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Interactive emoji suggester.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with per-emoji CSVs.")
    parser.add_argument("--n_rows", type=int, default=1000, help="Rows per CSV to learn from.")
    parser.add_argument("--text_col", type=str, default="text", help="Tweet text column name.")
    parser.add_argument("--top_k", type=int, default=1, help="How many emojis to print per query.")
    args = parser.parse_args()

    profiles = build_emoji_profiles(Path(args.data_dir), args.n_rows, args.text_col)
    if not profiles:
        print("No profiles built. Check data_dir and CSV schema.")
        return

    try:
        while True:
            user_in = input().strip()
            if not user_in:
                break
            results = best_emoji(user_in, profiles, top_k=args.top_k)
            print(" ".join([emj for _, emj in results]))
    except EOFError:
        pass


if __name__ == "__main__":
    main()
