import pandas as pd
import emoji
from collections import Counter, defaultdict
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- Load and prepare data ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv")

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
).apply(extract_emojis)

df = df[df["emoji_list"].apply(len) > 0].copy()
df["genre"] = df["categories"].fillna("").apply(lambda x: x.split(",")[0].strip())
df = df[df["genre"] != "['Movies & TV'"]


# --- Filter down to top genres ---
top_k = 5
top_genres = df["genre"].value_counts().nlargest(top_k).index
df = df[df["genre"].isin(top_genres)]

# --- Count emoji frequency per genre ---
genre_emoji_counts = defaultdict(Counter)

for _, row in df.iterrows():
    genre = row["genre"]
    emojis = set(row["emoji_list"])
    for e in emojis:
        genre_emoji_counts[genre][e] += 1

# --- Prepare DataFrame for plotting ---
rows = []
for genre in top_genres:
    for e, count in genre_emoji_counts[genre].items():
        rows.append((genre, e, count))

df_counts = pd.DataFrame(rows, columns=["genre", "emoji", "count"])
emoji_genre_spread = df_counts.groupby("emoji")["genre"].nunique()
df_counts = df_counts[df_counts["emoji"].isin(emoji_genre_spread[emoji_genre_spread <= 5].index)]

# Keep top N emojis per genre
top_n = 5
df_plot = (
    df_counts.sort_values("count", ascending=False)
    .groupby("genre")
    .head(top_n)
    .sort_values(["genre", "count"], ascending=[True, False])
)

# --- Plot ---
fig, axes = plt.subplots(len(top_genres), 1, figsize=(10, 2.5 * len(top_genres)))

emoji_font = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

for ax, genre in zip(axes, top_genres):
    subset = df_plot[df_plot["genre"] == genre]
    ax.barh(subset["emoji"], subset["count"], color="skyblue")
    ax.set_title(genre)
    ax.invert_yaxis()
    for label in ax.get_yticklabels():
        label.set_fontproperties(emoji_font)
        label.set_fontsize(14)
    ax.grid(True, axis="x")

plt.tight_layout()
plt.show()
