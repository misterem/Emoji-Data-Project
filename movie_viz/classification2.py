import math
import pandas as pd
import numpy as np
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import random
from nrclex import NRCLex

# --- Load and prepare dataset ---
df = pd.read_csv("../movie_data/reddit-amazon-emoji-only.csv")

def extract_emojis(text):
    return " ".join([c for c in str(text) if c in emoji.EMOJI_DATA])

df["emoji_str"] = (df["title_y"].fillna("") + " " + df["text"].fillna("")).apply(extract_emojis)

analyzer = SentimentIntensityAnalyzer()

# def classify_sentiment(text):
#     score = analyzer.polarity_scores(str(text))["compound"]
#     return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"

def classify_sentiment(text):
    emo = NRCLex(str(text))
    scores = emo.raw_emotion_scores
    if not scores:
        return None
    return max(scores, key=scores.get)  # dominant emotion

df["sentiment_class"] = df["text"].fillna("").apply(classify_sentiment)

df = df[df["emoji_str"].str.strip().astype(bool)].copy()

# --- Build emoji classifier ---
X = CountVectorizer(analyzer="char", token_pattern=None).fit_transform(df["emoji_str"])
y = df["sentiment_class"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# --- Build emoji sentiment dictionary ---
from collections import defaultdict, Counter

emoji_sentiment = defaultdict(Counter)

for _, row in df.iterrows():
    for e in set(row["emoji_str"].split()):
        emoji_sentiment[row["sentiment_class"]][e] += 1

# --- Function: suggest emoji for new text ---
def suggest_emoji_1(text):
    sentiment = classify_sentiment(text)
    emoji_counts = emoji_sentiment[sentiment]
    if not emoji_counts:
        return None
    return emoji_counts.most_common(1)[0][0]

def suggest_emoji_2(text, k=5):
    sentiment = classify_sentiment(text)
    emoji_counts = emoji_sentiment.get(sentiment, {})
    if not emoji_counts:
        return None
    top_k = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return random.choice([e for e, _ in top_k])

def suggest_emoji_3(text):
    sentiment = classify_sentiment(text)
    emoji_counts = emoji_sentiment.get(sentiment, {})
    if not emoji_counts:
        return None

    penalized = {e: count / math.log(count + 2) for e, count in emoji_counts.items()}
    return max(penalized, key=penalized.get)

def suggest_emoji_4(text):
    sentiment = classify_sentiment(text)
    emoji_counts = emoji_sentiment.get(sentiment, {})
    if not emoji_counts:
        return None
    emojis, weights = zip(*emoji_counts.items())
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(emojis, weights=probs, k=1)[0]


# --- Example usage ---
while(text_input := input("Enter text: ")):
    predicted_emoji = suggest_emoji_1(text_input)
    print(f"Suggested emoji: {predicted_emoji}")
    predicted_emoji = suggest_emoji_2(text_input)
    print(f"Suggested emoji: {predicted_emoji}")
    predicted_emoji = suggest_emoji_3(text_input)
    print(f"Suggested emoji: {predicted_emoji}")
    predicted_emoji = suggest_emoji_4(text_input)
    print(f"Suggested emoji: {predicted_emoji}")
