import pandas as pd
import emoji
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

# --- Load data and extract emojis ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv")

# --- Extract emojis from 'title_y' and 'text' ---
def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
).apply(extract_emojis)

all_emojis = df["emoji_list"].explode()
top5_emojis = Counter(all_emojis).most_common(5)

# --- Step 2: Generate word clouds ---
from wordcloud import WordCloud, STOPWORDS

custom_stopwords = STOPWORDS.union({"br", "movie"})

def generate_wordcloud(text):
    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        stopwords=custom_stopwords
    ).generate(text)

# --- Step 3: Plot side by side ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (emoji_char, _) in zip(axes, top5_emojis):
    emoji_df = df[df["emoji_list"].apply(lambda lst: emoji_char in lst)]
    text = " ".join(emoji_df["cleaned_text"].dropna().astype(str))
    wc = generate_wordcloud(text)

    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(emoji_char, fontsize=20)
    ax.axis("off")

plt.tight_layout()
plt.show()

# # --- Use 'cleaned_text' for word cloud content ---
# def plot_emoji_wordcloud(target_emoji):
#     emoji_df = df[df["emoji_list"].apply(lambda lst: target_emoji in lst)]
#     all_text = " ".join(emoji_df["cleaned_text"].dropna().astype(str).tolist())
#
#     custom_stopwords = STOPWORDS.union({"br", "movie"})
#     wc = WordCloud(width=1000, height=500, background_color="white", colormap="viridis", stopwords=custom_stopwords).generate(all_text)
#
#     plt.figure(figsize=(12, 6))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.title(f"Word Cloud for Reviews Containing '{target_emoji}'", fontsize=16)
#     plt.tight_layout()
#     plt.show()
#
# # --- Example ---
# plot_emoji_wordcloud("💔")
