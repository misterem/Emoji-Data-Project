import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json


HF_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_INDEX = {l: i for i, l in enumerate(HF_LABELS)}


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
    alias = f":{stem.strip().lower().replace(' ', '_').replace('-', '_') }:"
    emj = emoji.emojize(alias, language='alias')
    if emj != alias and emj in emoji.EMOJI_DATA:
        return emj
    counter = Counter()
    for t in texts.dropna().tolist():
        counter.update(_extract_emojis(t))
    return counter.most_common(1)[0][0] if counter else None


def build_emoji_profiles_hf(
    data_dir: Path,
    n_rows_per_file: int,
    text_column: str,
    batch_size: int,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    device: int | str = "cpu",
) -> dict:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        function_to_apply="softmax",
        device=device,
        truncation=True,
        max_length=256,
        return_all_scores=True,
    )

    profiles: dict[str, dict] = {}
    csvs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".csv"])
    csv_num = 0
    for p in csvs:
        csv_num += 1
        print(f"Processing csv {csv_num}/{len(csvs)}")
        try:
            df = pd.read_csv(p, nrows=n_rows_per_file)
        except Exception:
            continue

        if df.empty:
            continue

        if text_column not in df.columns:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            use_col = obj_cols[0] if obj_cols else df.columns[0]
        else:
            use_col = text_column

        emj = _infer_file_emoji(p.stem, df[use_col].head(min(len(df), 200)))
        if not emj:
            continue

        texts = df[use_col].fillna('').astype(str).tolist()
        probs_sum = np.zeros(len(HF_LABELS), dtype=float)
        cnt = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            out = clf(batch, top_k=None)
            for row in out:
                vec = np.zeros(len(HF_LABELS), dtype=float)
                for item in row:
                    idx = LABEL_INDEX.get(item["label"])
                    if idx is not None:
                        vec[idx] = float(item["score"])
                probs_sum += vec
                cnt += 1

        if cnt == 0:
            continue

        profiles[emj] = {"vec": _normalize(probs_sum), "count": cnt, "file": p.name}

    return profiles

def save_profiles(profiles: dict, path: str, meta: dict | None = None) -> None:
    """
    profiles: {emoji: {"vec": np.ndarray, "count": int, "file": str}}
    Saves compact JSON with float lists.
    """
    payload = {
        "meta": meta or {},
        "profiles": {
            emj: {
                "vec": d["vec"].tolist(),
                "count": int(d.get("count", 0)),
                "file": d.get("file", "")
            }
            for emj, d in profiles.items()
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_profiles(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prof = {}
    for emj, d in payload["profiles"].items():
        v = np.array(d["vec"], dtype=float)
        n = np.linalg.norm(v)
        v = v / n if n > 0 else v
        prof[emj] = {"vec": v, "count": int(d.get("count", 0)), "file": d.get("file", "")}
    return prof


def best_emoji(user_text: str, profiles: dict, clf, top_k: int = 1):
    out = clf([user_text], top_k=None)[0]
    v = np.zeros(len(HF_LABELS), dtype=float)
    for item in out:
        idx = LABEL_INDEX.get(item["label"])
        if idx is not None:
            v[idx] = float(item["score"])
    v = _normalize(v)

    scored = []
    for emj, d in profiles.items():
        s = _cosine(v, d["vec"])
        scored.append((s, emj))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def main():
    ap = argparse.ArgumentParser(description="Interactive emoji suggester using Hugging Face emotion model.")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with per-emoji CSVs.")
    ap.add_argument("--n_rows", type=int, default=1000, help="Rows per CSV to learn from.")
    ap.add_argument("--text_col", type=str, default="text", help="Tweet text column name.")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference.")
    ap.add_argument("--top_k", type=int, default=1, help="How many emojis to print per query.")
    ap.add_argument("--device", type=str, default="cpu", help="transformers device index, e.g., -1/cpu or 0 for CUDA")
    ap.add_argument("--model", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="HF model id")
    ap.add_argument("--profiles", type=str, default="emoji_profiles.json", help="Path to save/load cached emoji profiles.")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuilding profiles from CSVs and overwrite --profiles.")

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model)
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        function_to_apply="softmax",
        device=args.device,
        truncation=True,
        max_length=256,
        return_all_scores=True,
    )

    # Load cached profiles unless --rebuild or file missing
    if (not args.rebuild) and Path(args.profiles).exists():
        profiles = load_profiles(args.profiles)
    else:
        profiles = build_emoji_profiles_hf(
            Path(args.data_dir),
            n_rows_per_file=args.n_rows,
            text_column=args.text_col,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
        )
        if not profiles:
            print("No profiles built. Check data_dir and CSV schema.")
            return
        # Add minimal metadata; expand if needed
        meta = {
            "source_dir": str(Path(args.data_dir).resolve()),
            "n_rows_per_file": int(args.n_rows),
            "text_column": args.text_col,
            "model": "j-hartmann/emotion-english-distilroberta-base"
        }
        save_profiles(profiles, args.profiles, meta=meta)

    try:
        while True:
            user_in = input().strip()
            if not user_in:
                break
            results = best_emoji(user_in, profiles, clf, top_k=args.top_k)
            print(" ".join([emj for _, emj in results]))
    except EOFError:
        pass


if __name__ == "__main__":
    main()
