import os

import pandas as pd
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from collections import defaultdict, Counter
import random
from tqdm import tqdm

# --- Load and extract emojis ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv")


def extract_emojis(text):
    return [c for c in str(text) if c in emoji.EMOJI_DATA]


df["emoji_list"] = (
        df["title_y"].fillna("") + " " + df["text"].fillna("")
).apply(extract_emojis)

df = df[df["emoji_list"].apply(len) > 0].copy()

# --- Emotion classification using HuggingFace model ---
model_name = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

emotion_classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    top_k=1,
    truncation=True
)


def classify_emotion(text):
    return emotion_classifier(str(text))[0][0]["label"]


if os.path.exists("../movie_data/labeled_reviews.csv"):
    df = pd.read_csv("../movie_data/labeled_reviews.csv")

    df["emoji_list"] = (
            df["title_y"].fillna("") + " " + df["text"].fillna("")
    ).apply(extract_emojis)

    df = df[df["emoji_list"].apply(len) > 0].copy()
else:
    emotions = []
    for text in tqdm(df["text"].fillna(""), desc="Labeling reviews"):
        try:
            result = emotion_classifier(str(text))[0][0]
            emotions.append(result["label"])
        except Exception:
            emotions.append(None)

    df["emotion"] = emotions

    # df["emotion"] = df["text"].fillna("").apply(classify_emotion)
    df = df[df["emotion"].notna()]
    df.to_csv("labeled_reviews.csv", index=False)

# --- Build emoji-emotion frequency mapping ---
emoji_emotion_counts = defaultdict(Counter)

for _, row in df.iterrows():
    emotion = row["emotion"]
    for e in set(row["emoji_list"]):
        emoji_emotion_counts[emotion][e] += 1


# --- Suggestion function with optional filtering ---
def suggest_emoji_from_emotion(text, top_k=3, exclude={"👍", "👎", "🏿", "🏾", "🏻", "🏼", "🏽"}):
    emotion = classify_emotion(text)
    if not emotion:
        return None
    counts = {e: c for e, c in emoji_emotion_counts[emotion].items() if e not in exclude}
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return random.choice([e for e, _ in top])


# --- Example usage ---
text_input = "I'm really disappointed with how the meeting went."
predicted_emoji = suggest_emoji_from_emotion(text_input)
print(f"Suggested emoji for '{classify_emotion(text_input)}': {predicted_emoji}")

while (text_input := input("Enter text: ")):
    predicted_emoji = suggest_emoji_from_emotion(text_input)
    print(f"Suggested emoji for '{classify_emotion(text_input)}': {predicted_emoji}")
