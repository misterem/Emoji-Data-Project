import pandas as pd
import numpy as np
import emoji
from collections import Counter, defaultdict
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- Step 1: Load and extract emojis ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv", encoding="utf-8")

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
).apply(extract_emojis)

df = df[df["emoji_list"].apply(len) > 0].copy()

# --- Step 2: Extract primary genre from 'categories' or 'features' ---
# Adjust column name based on actual format
# df["genre"] = df["categories"].fillna("").apply(lambda x: x.split(",")[0].strip() if "," in x else x.strip())
df["genre"] = df["categories"].fillna("").apply(lambda x: x.split(",")[0].strip())

top_k = 10  # or any reasonable number
top_genres = df["genre"].value_counts().nlargest(top_k).index
df = df[df["genre"].isin(top_genres)]

# --- Step 3: Build emoji-genre frequency matrix ---
emoji_genre_counts = defaultdict(Counter)

for _, row in df.iterrows():
    genre = row["genre"]
    emojis = set(row["emoji_list"])
    for e in emojis:
        emoji_genre_counts[e][genre] += 1

# Create frequency table
emoji_list = list(emoji_genre_counts.keys())
genre_list = list({g for counts in emoji_genre_counts.values() for g in counts})
freq_matrix = pd.DataFrame(index=emoji_list, columns=genre_list).fillna(0)

for e in emoji_list:
    for g in emoji_genre_counts[e]:
        freq_matrix.loc[e, g] = emoji_genre_counts[e][g]

freq_matrix.drop(columns=["['Movies & TV'"], errors="ignore", inplace=True)
freq_matrix.drop(index=["🏿", "🏾", "🏻", "🏼", "🏽", "♂"], errors="ignore", inplace=True)
# --- Step 4: Chi-square test to detect emoji-genre association ---
chi2_vals = {}
for e in freq_matrix.index:
    obs = freq_matrix.loc[e].values
    if obs.sum() < 10:
        continue
    expected = np.outer(obs.sum(), freq_matrix.sum(axis=0) / freq_matrix.values.sum())
    chi2 = ((obs - expected) ** 2 / expected).sum()
    chi2_vals[e] = chi2

# Select top N associated emojis
top_emojis = sorted(chi2_vals, key=chi2_vals.get, reverse=True)[:10]
for e in top_emojis:
    print(repr(e))
# --- Step 5: Visualization ---
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
subset = freq_matrix.loc[top_emojis]
subset_norm = subset.div(subset.sum(axis=1), axis=0)  # normalize per emoji
subset_norm.plot(kind="barh", stacked=True, figsize=(12, 6), colormap="tab20")
plt.xlabel("Proportion of Genre Mentions")
plt.ylabel("Emoji")
plt.title("Top Emojis with Strong Genre Association (Chi² Selection)")
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(14)  # optional
plt.tight_layout()
plt.show()
