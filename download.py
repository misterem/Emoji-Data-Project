import numpy as np
from datasets import load_dataset
import pandas as pd
import re
import emoji

# Load the dataset
dataset = load_dataset("rohan2810/amazon-reddit-merged-matched-reviews-all", split="train")

# def contains_emoji(text):
#     return any(char in emoji.EMOJI_DATA for char in text)
#
# # Function to detect if a text contains any emoji using regex
# # emoji_pattern = re.compile(
# #     "["
# #     "\U0001F600-\U0001F64F"  # emoticons
# #     "\U0001F300-\U0001F5FF"  # symbols & pictographs
# #     "\U0001F680-\U0001F6FF"  # transport & map symbols
# #     "\U0001F1E0-\U0001F1FF"  # flags
# #     "\U00002700-\U000027BF"  # dingbats
# #     "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
# #     "\U00002600-\U000026FF"  # miscellaneous symbols
# #     "\U0001FA70-\U0001FAFF"  # extended symbols (e.g. hearts, hands)
# #     "\U000025A0-\U000025FF"  # geometric shapes
# #     "]", flags=re.UNICODE)
#
# # Filter reviews that contain emojis
# # rows_with_emojis = [row for row in dataset if re.search(emoji_pattern, row["text"])]
# rows_with_emojis = [row for row in dataset if contains_emoji(["text"])]
# df_with_emojis = pd.DataFrame(rows_with_emojis)
#
# # Drop duplicate asin values (keep first occurrence)
# df_unique_asin = df_with_emojis.drop_duplicates(subset='asin')
#
# # Save to CSV
# df_unique_asin.to_csv('emoji_reviews.csv', index=False)
#
# print(f"Saved {len(df_unique_asin)} reviews with emojis and unique ASINs.")
# Use Hugging Face's `filter` to keep only emoji-containing reviews
def has_emoji(example):
    return bool(emoji.emoji_list(example['text'])) or bool(emoji.emoji_list(example['title']))

filtered_dataset = dataset.filter(has_emoji)

# Convert to pandas DataFrame
df = filtered_dataset.to_pandas()

# Drop duplicate 'asin' values
# df_unique = df.drop_duplicates()
hashable_columns = [col for col in df.columns if not isinstance(df[col].iloc[0], (list, dict, np.ndarray))]

# Drop duplicates using only hashable columns but keep all data
df_unique = df.loc[df[hashable_columns].drop_duplicates().index]

# Save to CSV
df_unique.to_csv("emoji_reviews.csv", index=False)

print(f"Saved {len(df_unique)} emoji-containing, unique-asin reviews to emoji_reviews.csv")
