# features.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack


# Load Data

df = pd.read_csv("clean_data.csv", encoding="utf-8")

# Handle any NaN
df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)


# 1. Feature Engineering 
def count_punct(text):
    return sum([1 for char in str(text) if char in "!?.،؛"])

# Feature: Tweet Length
df['len'] = df['text'].apply(lambda x: len(str(x)))

# Feature: Punctuation Count
df['punct'] = df['text'].apply(count_punct)

# Feature: Word Count
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# Normalize Manual Features (0-1) for Naive Bayes
scaler = MinMaxScaler()
X_manual = scaler.fit_transform(df[['len', 'punct', 'word_count']])


# 2. TF-IDF Features (Content-Based)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(df["clean_text"])


# 3. Combine Features

X_final = hstack([X_tfidf, X_manual])
y = df["label"]


# 4. Split Data (60-20-20)
# First split: 60% Train, 40% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y, test_size=0.4, random_state=42, stratify=y
)

# Second split: 20% Val, 20% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# Save
with open("data_splits.pkl", "wb") as f:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Feature Extraction Complete")
print(f"Total Features: {X_final.shape[1]}")
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")