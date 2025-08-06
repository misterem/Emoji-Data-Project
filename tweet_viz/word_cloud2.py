import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Load CSV
df = pd.read_csv("../archive/ghost.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", '', text)  # Remove punctuation/numbers
    return text

# Apply cleaning
df['clean_text'] = df['Text'].fillna('').apply(clean_text)

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [word for sentence in df['clean_text'] for word in sentence.split() if word not in stop_words]

# Count frequencies
word_freq = Counter(tokens)

# Generate word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# Display
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()

