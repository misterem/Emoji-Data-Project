import os
import pandas as pd
import emoji
import seaborn as sns
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
from itertools import combinations
from tqdm import tqdm
import numpy as np

# Get all CSV files
folder = "../archive"
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

# Map file names to emojis (assumes file name like loudly_crying_face.csv for 😭)
emoji_map = {
    "backhand_index_pointing_right": "👉",
    "check_mark": "✔",
    "check_mark_button": "✅",
    "clown_face": "🤡",
    "cooking": "🍳",
    "egg": "🥚",
    "enraged_face": "😡",
    "eyes": "👀",
    "face_holding_back_tears": "🥹",
    "face_savoring_food": "😋",
    "face_with_steam_from_nose": "😤",
    "face_with_tears_of_joy": "😂",
    "fearful_face": "😨",
    "fire": "🔥",
    "folded_hands": "🙏",
    "ghost": "👻",
    "grinning_face_with_sweat": "😅",
    "hatching_chick": "🐣",
    "hot_face": "🥵",
    "loudly_crying_face": "😭",
    "melting_face": "🫠",
    "middle_finger": "🖕",
    "party_popper": "🎉",
    "partying_face": "🥳",
    "pile_of_poo": "💩",
    "rabbit": "🐇",
    "rabbit_face": "🐰",
    "red_heart": "❤️",
    "rolling_on_the_floor_laughing": "🤣",
    "saluting_face": "🫡",
    "skull": "💀",
    "smiling_face": "🙂",
    "smiling_face_with_halo": "😇",
    "smiling_face_with_heart-eyes": "😍",
    "smiling_face_with_hearts": "🥰",
    "smiling_face_with_sunglasses": "😎",
    "smiling_face_with_tear": "🥲",
    "sparkles": "✨",
    "sun": "☀️",
    "thinking_face": "🤔",
    "thumbs_up": "👍",
    "white_heart": "🤍",
    "winking_face": "😉"
}


# Function to extract emojis from text
def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

# Count co-occurrences
pair_counts = Counter()

for file in tqdm(files):
    emoji_label = os.path.splitext(file)[0]
    if emoji_label not in emoji_map:
        continue
    df = pd.read_csv(os.path.join(folder, file), on_bad_lines='skip', encoding='utf-8', engine='python')
    texts = df["Text"].fillna("").astype(str)
    for text in texts:
        found = set(extract_emojis(text))
        if len(found) > 1:
            for pair in combinations(sorted(found), 2):
                pair_counts[pair] += 1

# Get all unique emojis
unique_emojis = sorted(set([e for pair in pair_counts for e in pair]))

# Build co-occurrence matrix
matrix = pd.DataFrame(0, index=unique_emojis, columns=unique_emojis)

for (e1, e2), count in pair_counts.items():
    matrix.at[e1, e2] = count
    matrix.at[e2, e1] = count  # Symmetric

# Plot heatmap
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# Get top 20 co-occurrences
top_pairs = pair_counts.most_common(20)

# Extract involved emojis
top_emojis = sorted(set(e for pair, _ in top_pairs for e in pair))

# Build reduced matrix
reduced = matrix.loc[top_emojis, top_emojis]

logged = np.log1p(reduced)
normed = (logged - logged.min().min()) / (logged.max().max() - logged.min().min())
# Plot
sns.heatmap(normed, cmap="Reds", square=True, linewidths=0.5, annot=reduced, fmt="d", annot_kws={"size": 4})
plt.title("Top 20 Emoji Co-occurrence Heatmap")
plt.xticks(fontproperties=prop)
plt.yticks(fontproperties=prop)
plt.tight_layout()
# plt.show()
plt.savefig('tweet_co-occurrences', dpi=300)