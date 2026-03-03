import pandas as pd
import emoji
import ast
import plotly.graph_objects as go

df = pd.read_csv("data/reddit-reviews-emojis-clean.csv", low_memory=False)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["emojis"] = df["text"].apply(emoji.distinct_emoji_list)
df["categories"] = df["categories"].apply(ast.literal_eval)

# compute baseline rating
df_baseline = df.copy()
df_baseline = (
    df_baseline.explode("emojis")
    .groupby("emojis")["rating"]
    .agg(["mean", "size"])
    .rename(columns={"mean": "baseline_rating", "size": "total_count"})
)
df_baseline.index.name = "emojis"

# compute rating per category
df_cat = df[(df["emojis"].map(len) > 0) & (df["categories"].map(len) > 0)].copy()
df_cat = (
    df_cat.explode("categories")
    .explode("emojis")
    .groupby(["emojis", "categories"])["rating"]
    .agg(["mean", "size"])
    .rename(columns={"mean": "category_rating", "size": "category_count"})
)

# Join the category-specific data with the baseline data
df_final = df_cat.join(df_baseline, on="emojis")

# filter categories with low support
MIN_OCCURRENCES = 20
df_final = df_final[df_final["category_count"] >= MIN_OCCURRENCES]
df_final["rating_shift"] = df_final["category_rating"] - df_final["baseline_rating"]

print(f"--- Top 10 Emojis That Become MORE POSITIVE ---")
print(
    df_final.sort_values(by="rating_shift", ascending=False)
    .head(10)
    .to_string(
        formatters={
            "category_rating": "{:.2f}".format,
            "baseline_rating": "{:.2f}".format,
            "rating_shift": "{:+.2f}".format,
        }
    )
)

print(f"\n--- Top 10 Emojis That Become MORE NEGATIVE ---")
print(
    df_final.sort_values(by="rating_shift", ascending=True)
    .head(10)
    .to_string(
        formatters={
            "category_rating": "{:.2f}".format,
            "baseline_rating": "{:.2f}".format,
            "rating_shift": "{:+.2f}".format,
        }
    )
)

df_final.to_csv("emoji-rating-shift-by-category.csv")

df_plt = df_final.reset_index()

# plot genres
for genre in ["Drama", "Thriller"]:
    df_genre = df_plt[df_plt["categories"] == genre]
    df_genre["diff"] = df_genre["baseline_rating"] - df_genre["category_rating"]
    df_genre["sqdiff"] = df_genre["diff"] ** 2
    df_genre = df_genre.sort_values(by="sqdiff", ascending=False).head(10)
    df_genre.sort_values(by="diff", inplace=True)
    fig = go.Figure(
        data=[
            go.Bar(
                name="Baseline",
                x=df_genre["emojis"],
                y=df_genre["baseline_rating"],
                marker_color="blue",
            ),
            go.Bar(
                name="Category",
                x=df_genre["emojis"],
                y=df_genre["category_rating"],
                marker_color="green",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"Emoji rating - {genre} Movies vs. Global Average",
        xaxis_tickfont=dict(size=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.show()
