import csv
from typing import Any
from emoji import emoji_count
from tqdm import tqdm
from datasets import load_dataset

ds: list[dict[str, Any]] = load_dataset("rohan2810/amazon-reddit-merged-matched-reviews-all", split="train")
row = ds[0]

with_emojis, total = 0, 0
with open("movie_data/reddit-amazon-emoji-only.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=ds[0].keys())
    writer.writeheader()
    for row in tqdm(ds, total=5330000):
        total += 1
        if emoji_count(row["text"] + row["title_y"]) > 0 and row["title_y"] != last_row["title_y"]:
            with_emojis += 1
            writer.writerow(row)
        last_row = row

print(f"{with_emojis}/{total} (={with_emojis / total * 100:.2f}%)")
