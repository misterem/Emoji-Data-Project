import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import emoji
from nrclex import NRCLex

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


def best_emoji(user_text: str, profiles: dict, top_k: int = 1):
    """Return top_k emojis ranked by cosine similarity to user's NRCLex vector."""
    v = _normalize(_nrc_vector(user_text))
    scored = []
    for emj, d in profiles.items():
        s = _cosine(v, d["vec"])
        scored.append((s, emj))
    scored.sort(reverse=True, key=lambda x: x[0])
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

    # REPL: user enters a sentence; program prints the most fitting emoji(s)
    try:
        while True:
            user_in = input().strip()
            if not user_in:
                break
            results = best_emoji(user_in, profiles, top_k=args.top_k)
            # Print only emojis, space-separated, highest first
            print(" ".join([emj for _, emj in results]))
    except EOFError:
        pass


if __name__ == "__main__":
    main()
