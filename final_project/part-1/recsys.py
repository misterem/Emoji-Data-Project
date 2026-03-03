import pandas as pd
import numpy as np
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import plotly.express as px

df = pd.read_csv("data/reddit-reviews-emojis-clean.csv")
df.dropna(subset=["text", "rating"], inplace=True)

df["emojis"] = df["text"].apply(emoji.distinct_emoji_list)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=0, stratify=df["rating"]
)

all_emojis = sorted({e for review in train_df["emojis"] for e in review})
emoji_to_idx = {emoji: i for i, emoji in enumerate(all_emojis)}

# emoji profiles - each column i represents reviews of (i+1) stars
profiles = np.zeros((len(all_emojis), 5))

for _, row in train_df.iterrows():
    for e in row["emojis"]:
        profiles[emoji_to_idx[e]][row["rating"] - 1] += 1

profiles /= profiles.sum(axis=1, keepdims=True)

GLOBAL_AVG_RATING = train_df["rating"].mean()


def predict(emojis_in_review):
    known_emojis = [e for e in emojis_in_review if e in emoji_to_idx]

    # fallback - we didn't train on any of the emojis
    if not known_emojis:
        return GLOBAL_AVG_RATING

    indices = [emoji_to_idx[e] for e in known_emojis]
    combined_profile = profiles[indices].sum(axis=0)

    final_distribution = combined_profile / len(known_emojis)
    return np.sum(final_distribution * np.arange(1, 6))


# predict, evaluate and compare to best guess
predictions = test_df["emojis"].apply(predict)
rmse = np.sqrt(mean_squared_error(test_df["rating"], predictions))
print(f"Model Performance (RMSE): {rmse:.4f}")

baseline_rmse = np.sqrt(
    mean_squared_error(
        test_df["rating"], GLOBAL_AVG_RATING * np.ones_like(test_df["rating"])
    )
)
print(f"Baseline - Best Guess Performance (RMSE): {baseline_rmse:.4f}")

# compute error per emoji
sq_err = (test_df["rating"] - predictions) ** 2
emoji_err = defaultdict(lambda: 0)
for i, err in zip(sq_err.index, sq_err):
    for e in df.iloc[i]["emojis"]:
        emoji_err[e] += err
emoji_err = pd.DataFrame(emoji_err, index=["error"]).transpose()
emoji_err.sort_values(by="error", inplace=True, ascending=False)

worst_errors = emoji_err.head(20)
fig = px.bar(
    worst_errors,
    x=worst_errors.index,
    y="error",
    labels={"x": "Emoji", "error": "Squared Error"},
    title="Squared Error Per Emoji",
)
fig.update_layout(
    xaxis_tickfont=dict(size=24)  # increase as needed
)
fig.show()
