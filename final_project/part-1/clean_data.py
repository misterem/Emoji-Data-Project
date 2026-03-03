import ast
import pandas as pd


df = pd.read_csv("data/reddit-reviews-emojis.csv")
category_map = pd.read_csv("part-1/category_mapping.csv").set_index(
    "original_category"
)["clean_category"]


def clean_categories(categories: str) -> frozenset[str]:
    # categories in the dataset are present as string repr of python lists
    parsed = ast.literal_eval(categories)
    # use frozenset for hashable (for dedup), unique values
    return frozenset(category_map.reindex(parsed).dropna())


# pick only relevant columns
df = df[["title", "rating", "title_y", "text", "categories"]].copy()
df = df.rename(columns={"title": "movie"})
# combine title + content into a single review column
df["text"] = df["title_y"] + " " + df["text"]
df.dropna(subset=["text"], inplace=True)
df.drop(columns=["title_y"], inplace=True)
# parse rating
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df.dropna(subset=["rating"], inplace=True)
df["rating"] = df["rating"].astype(int)
# clean categories
df["categories"] = df["categories"].apply(clean_categories)
# dedup
df.drop_duplicates(inplace=True)
# export
df["categories"] = df["categories"].apply(sorted)
df.to_csv("data/reddit-reviews-emojis-clean.csv", index=False)
