import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk.corpus import stopwords

emoji = "skull"
# Load CSV
df = pd.read_csv(f"../archive/{emoji}.csv")

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
stop_words.update(['gt', 'lt', 'amp'])
tokens = [word for sentence in df['clean_text'] for word in sentence.split() if word not in stop_words]

# Count frequencies
word_freq = Counter(tokens)

# Generate word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# Display
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.title(f"Most common words for {emoji} emoji", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.show()

