import argparse
import os
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import emoji
from nrclex import NRCLex


def extract_emojis(s: str):
    """Return list of emoji characters found in s."""
    return [c for c in str(s) if c in emoji.EMOJI_DATA]


def guess_emoji_for_file(stem: str, texts: pd.Series) -> str | None:
    """
    Infer the emoji represented by this file.
    1) Try emoji.emojize on the filename stem (underscored).
    2) Fallback: most common emoji found in the sample texts.
    """
    alias = f":{stem.strip().lower().replace(' ', '_').replace('-', '_')}:"
    emj = emoji.emojize(alias, language='alias')
    if emj != alias and emj in emoji.EMOJI_DATA:
        return emj

    counter = Counter()
    for t in texts.dropna().tolist():
        counter.update(extract_emojis(t))
    if counter:
        return counter.most_common(1)[0][0]
    return None


def sentiment_scores(text: str) -> dict[str, float]:
    """
    NRCLex raw_emotion_scores includes 'positive' and 'negative'.
    Return only those two.
    """
    if not isinstance(text, str) or not text:
        return {'positive': 0.0, 'negative': 0.0}
    lex = NRCLex(text)
    scores = lex.raw_emotion_scores
    pos = float(scores.get('positive', 0.0))
    neg = float(scores.get('negative', 0.0))
    return {'positive': pos, 'negative': neg}


def process_directory(data_dir: str,
                      n_rows_per_file: int = 1000,
                      text_column: str = 'text') -> pd.DataFrame:
    """
    Read first n_rows_per_file rows from each CSV in data_dir, compute aggregate pos/neg per emoji.
    Returns columns:
    ['emoji','file','n_rows','pos_sum','neg_sum','pos_mean','neg_mean','pos_ratio','neg_ratio','net_score']
    """
    records = []
    data_path = Path(data_dir)
    csvs = sorted([p for p in data_path.iterdir() if p.suffix.lower() == '.csv'])
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=n_rows_per_file)
        except Exception:
            continue

        if text_column not in df.columns:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            use_col = obj_cols[0] if obj_cols else df.columns[0]
        else:
            use_col = text_column

        emj = guess_emoji_for_file(p.stem, df[use_col].head(min(len(df), 200)))

        pos_total = 0.0
        neg_total = 0.0
        valid = 0
        for txt in df[use_col].fillna(''):
            sc = sentiment_scores(txt)
            pos_total += sc['positive']
            neg_total += sc['negative']
            valid += 1

        if valid == 0:
            continue

        pos_mean = pos_total / valid
        neg_mean = neg_total / valid
        denom = pos_total + neg_total
        pos_ratio = (pos_total / denom) if denom > 0 else 0.0
        neg_ratio = (neg_total / denom) if denom > 0 else 0.0
        net_score = pos_total - neg_total

        records.append({
            'emoji': emj,
            'file': p.name,
            'n_rows': valid,
            'pos_sum': pos_total,
            'neg_sum': neg_total,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'net_score': net_score
        })

    return pd.DataFrame.from_records(records)


def choose_top_emojis(summary_df: pd.DataFrame, top_k: int = 10) -> dict:
    """
    Return dict with top_k for 'positive' and 'negative'.
    Uses net_score for ordering; reverse for negative.
    """
    df = summary_df.copy()
    df = df[df['emoji'].notna()]

    pos_top = df.sort_values(['net_score', 'pos_ratio', 'pos_sum'], ascending=[False, False, False]).head(top_k)
    neg_top = df.sort_values(['net_score', 'neg_ratio', 'neg_sum'], ascending=[True, False, False]).head(top_k)

    return {
        'positive': [{'emoji': r.emoji, 'file': r.file, 'net_score': r.net_score, 'pos_ratio': r.pos_ratio}
                     for r in pos_top.itertuples(index=False)],
        'negative': [{'emoji': r.emoji, 'file': r.file, 'net_score': r.net_score, 'neg_ratio': r.neg_ratio}
                     for r in neg_top.itertuples(index=False)]
    }


def main():
    parser = argparse.ArgumentParser(description='Select best emojis for each sentiment from per-emoji CSVs.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing per-emoji CSV files.')
    parser.add_argument('--n_rows', type=int, default=1000, help='Rows to read from each CSV.')
    parser.add_argument('--text_col', type=str, default='text', help='Name of the tweet text column.')
    parser.add_argument('--top_k', type=int, default=10, help='How many emojis to return for each sentiment.')
    parser.add_argument('--out_csv', type=str, default='emoji_sentiment_summary.csv', help='Summary CSV path.')
    args = parser.parse_args()

    summary = process_directory(args.data_dir, n_rows_per_file=args.n_rows, text_column=args.text_col)
    summary.sort_values('net_score', ascending=False).to_csv(args.out_csv, index=False)

    top = choose_top_emojis(summary, top_k=args.top_k)

    def fmt(lst):
        return ', '.join([f"{Path(d['file']).stem}" for d in lst])

    print('Top positive:', fmt(top['positive']))
    print('Top negative:', fmt(top['negative']))


if __name__ == '__main__':
    main()
