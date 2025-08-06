import pandas as pd
import emoji
from nrclex import NRCLex
from collections import defaultdict, Counter
import random

# --- Load and extract emojis ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv")

def extract_emojis(text):
    return [c for c in str(text) if c in emoji.EMOJI_DATA]

df["emoji_list"] = (
    df["title_y"].fillna("") + " " + df["text"].fillna("")
).apply(extract_emojis)

df = df[df["emoji_list"].apply(len) > 0].copy()

# --- Emotion classification using NRCLex ---
def classify_emotion(text):
    emo = NRCLex(str(text))
    scores = emo.top_emotions
    print(emo.top_emotions)
    if not scores:
        return None
    return max(score for score in scores)
    #return emo.top_emotions[0]

df["emotion"] = df["text"].fillna("").apply(classify_emotion)
df = df[df["emotion"].notna()]

# --- Build emoji-emotion frequency mapping ---
emoji_emotion_counts = defaultdict(Counter)

for _, row in df.iterrows():
    emotion = row["emotion"]
    for e in set(row["emoji_list"]):
        emoji_emotion_counts[emotion][e] += 1

# --- Suggestion function with optional filtering ---
def suggest_emoji_from_emotion(text, top_k=3, exclude={"👍", "👎"}):
    emotion = classify_emotion(text)
    if not emotion:
        return None
    counts = {e: c for e, c in emoji_emotion_counts[emotion].items() if e not in exclude}
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return random.choice([e for e, _ in top])

# --- Example usage ---
while(text_input := input("Enter text: ")):
    predicted_emoji = suggest_emoji_from_emotion(text_input)
    print(f"Suggested emoji for '{classify_emotion(text_input)}': {predicted_emoji}")

