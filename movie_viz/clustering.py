import pandas as pd
import numpy as np
import emoji
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from collections import Counter

prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
# --- Step 1: Load preprocessed emoji review dataset ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv", encoding="utf-8")

# --- Step 2: Extract emojis from 'title_y' and 'text' using emoji package ---
def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
).apply(extract_emojis)

# --- Step 3: Filter rows with at least one emoji ---
df = df[df["emoji_list"].apply(len) > 0].copy()

# Initialize counters
emoji_review_counter = defaultdict(set)
emoji_movie_counter = defaultdict(set)
emoji_cooccur_counts = defaultdict(list)

# Loop through reviews
for _, row in df.iterrows():
    emojis = row["emoji_list"]
    review_id = row.name
    movie_id = row.get("amazon_name", row.get("reddit_movie"))

    unique_emojis = set(emojis)
    for e in unique_emojis:
        emoji_review_counter[e].add(review_id)
        emoji_movie_counter[e].add(movie_id)
        # Co-occurring emojis (excluding itself)
        cooccur = unique_emojis - {e}
        emoji_cooccur_counts[e].append(len(cooccur))

# --- Step 4: Aggregate statistics per emoji based on 'rating' column ---
emoji_rating_map = defaultdict(list)

for _, row in df.iterrows():
    for e in row["emoji_list"]:
        emoji_rating_map[e].append(row["rating"])

emoji_stats = pd.DataFrame({
    "emoji": list(emoji_rating_map.keys()),
    "count": [len(ratings) for ratings in emoji_rating_map.values()],
    "avg_rating": [np.mean(ratings) for ratings in emoji_rating_map.values()],
    "std_rating": [np.std(ratings) for ratings in emoji_rating_map.values()]
})
emoji_stats["review_count"] = emoji_stats["emoji"].map(lambda e: len(emoji_review_counter[e]))
emoji_stats["movie_count"] = emoji_stats["emoji"].map(lambda e: len(emoji_movie_counter[e]))
emoji_stats["avg_cooccurring_emojis"] = emoji_stats["emoji"].map(lambda e: np.mean(emoji_cooccur_counts[e]) if emoji_cooccur_counts[e] else 0)
emoji_stats["unique_cooccurring_emojis"] = emoji_stats["emoji"].map(lambda e: len(set().union(*[set(emoji_list) - {e} for emoji_list in df[df["emoji_list"].apply(lambda l: e in l)]["emoji_list"]])))


# --- Step 5: Feature normalization ---
# features = emoji_stats[["count", "avg_rating", "std_rating"]]
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
feature_cols = ["count", "avg_rating", "std_rating", "review_count", "movie_count", "avg_cooccurring_emojis", "unique_cooccurring_emojis"]
features = emoji_stats[feature_cols]
features_scaled = StandardScaler().fit_transform(features)

# --- Step 6: KMeans clustering ---
n_clusters = 5  # tunable
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
emoji_stats["cluster"] = kmeans.fit_predict(features_scaled)

# --- Step 7: Dimensionality reduction (PCA) for visualization ---
pca = PCA(n_components=2)
coords = pca.fit_transform(features_scaled)
emoji_stats["pca1"] = coords[:, 0]
emoji_stats["pca2"] = coords[:, 1]

# --- Step 8: Plot ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=emoji_stats, x="pca1", y="pca2", hue="cluster", s=100)

for _, row in emoji_stats.iterrows():
    if row["count"] > 10:
        plt.text(row["pca1"] + 0.05, row["pca2"], row["emoji"], fontsize=12, fontproperties=prop)

plt.title("Emoji Clusters Based on Review Rating Behavior")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
