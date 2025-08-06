import pandas as pd
import numpy as np
import emoji
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# --- Step 1: Load and prepare data ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv", encoding="utf-8")

# def extract_emojis(text):
#     return [char for char in text if char in emoji.EMOJI_DATA]
#
# df["emoji_list"] = (
#     df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
# ).apply(extract_emojis)
#
# df = df[df["emoji_list"].apply(len) > 0].copy()
#
# # --- Sentiment analysis ---
# analyzer = SentimentIntensityAnalyzer()
# df["sentiment_dict"] = df["text"].fillna("").apply(analyzer.polarity_scores)
#
# # --- Aggregate positive and negative sentiment per emoji ---
# emoji_pos = defaultdict(list)
# emoji_neg = defaultdict(list)
#
# for _, row in df.iterrows():
#     s = row["sentiment_dict"]
#     for e in set(row["emoji_list"]):
#         emoji_pos[e].append(s["pos"])
#         emoji_neg[e].append(s["neg"])
#
# emoji_scores = pd.DataFrame({
#     "emoji": list(emoji_pos.keys()),
#     "avg_pos": [np.mean(emoji_pos[e]) for e in emoji_pos],
#     "avg_neg": [np.mean(emoji_neg[e]) for e in emoji_neg],
#     "count": [len(emoji_pos[e]) for e in emoji_pos]
# })
#
# # --- Filter ---
# emoji_scores = emoji_scores[emoji_scores["count"] >= 50].copy()
# emoji_scores["pos_share"] = emoji_scores["avg_pos"] / (emoji_scores["avg_pos"] + emoji_scores["avg_neg"])
# emoji_scores["neg_share"] = 1 - emoji_scores["pos_share"]
#
# # --- Select top N by absolute polarity split ---
# emoji_scores["polarity_gap"] = abs(emoji_scores["pos_share"] - 0.5)
# top_emojis = emoji_scores.sort_values("polarity_gap", ascending=False).head(6).sort_values("pos_share")
# bot_emojis = emoji_scores.sort_values("polarity_gap", ascending=False).tail(6).sort_values("pos_share")
#
# # --- Plot ---
# prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
# fig, ax = plt.subplots(figsize=(10, 6))
# barh_neg = ax.barh(bot_emojis["emoji"], -bot_emojis["neg_share"], color="#FF7043", label="negative")
# barh_pos = ax.barh(top_emojis["emoji"], top_emojis["pos_share"], color="#4A6CD4", label="positive")
#
# for bar in barh_neg:
#     width = bar.get_width()
#     ax.text(width - 0.02, bar.get_y() + bar.get_height()/2, f"{int(width * 100)}%", va="center", ha="right", fontsize=9, fontproperties=prop)
#
# for bar in barh_pos:
#     width = bar.get_width()
#     ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{int(width * 100)}%", va="center", ha="left", fontsize=9, fontproperties=prop)
#
# ax.set_xlim(-1, 1)
# ax.set_xticks(np.linspace(-1, 1, 9))
# ax.set_xticklabels([f"{int(abs(x)*100)}%" for x in np.linspace(-1, 1, 9)])
# ax.set_xlabel("% of Sentiment")
# ax.set_title("Share of Positive vs. Negative Sentiment by Emoji")
# ax.legend(loc="lower right")
# plt.axvline(0, color="black", linewidth=0.5)
# plt.tight_layout()
# for label in ax.get_yticklabels():
#     label.set_fontproperties(prop)
#     label.set_fontsize(14)
# plt.show()
def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
).apply(extract_emojis)

df = df[df["emoji_list"].apply(len) > 0].copy()

# --- Step 2: Compute sentiment for each review ---
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["text"].fillna("").apply(lambda x: analyzer.polarity_scores(x)["compound"])

# --- Step 3: Aggregate sentiment per emoji ---
emoji_sentiment_map = defaultdict(list)

for _, row in df.iterrows():
    sentiment = row["sentiment"]
    for e in set(row["emoji_list"]):
        emoji_sentiment_map[e].append(sentiment)

emoji_sentiments = pd.DataFrame({
    "emoji": list(emoji_sentiment_map.keys()),
    "avg_sentiment": [np.mean(v) for v in emoji_sentiment_map.values()],
    "count": [len(v) for v in emoji_sentiment_map.values()]
})

# --- Step 4: Filter for frequently used emojis ---
emoji_sentiments = emoji_sentiments[emoji_sentiments["count"] >= 1]

# --- Step 5: Plot top positive and negative emojis ---
top_pos = emoji_sentiments.sort_values("avg_sentiment", ascending=False).head(10)
top_neg = emoji_sentiments.sort_values("avg_sentiment").head(10)
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].bar(top_pos["emoji"], top_pos["avg_sentiment"], color="green")
axs[0].set_title("Most Positive Emojis (VADER)", fontsize=14)
axs[0].set_ylim(-1, 1)

axs[1].bar(top_neg["emoji"], top_neg["avg_sentiment"], color="red")
axs[1].set_title("Most Negative Emojis (VADER)", fontsize=14)
axs[1].set_ylim(-1, 1)

for ax in axs:
    for label in ax.get_xticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(14)
    ax.set_xlabel("Emoji")
    ax.set_ylabel("Average Sentiment")
    ax.grid(True)

plt.tight_layout()
plt.show()
