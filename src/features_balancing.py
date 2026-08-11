# features_balancing.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy.sparse import hstack

# 1. Load Data
df = pd.read_csv("clean_data.csv", encoding="utf-8")
df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)

# 2. Split Data 
# Split: 60% Train, 20% Val, 20% Test
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])


# 3. Balance Training Data (Upsampling)
# Separate classes
df_neutral = train_df[train_df['label'] == 'NEUTRAL']
df_neg = train_df[train_df['label'] == 'NEG']
df_pos = train_df[train_df['label'] == 'POS']

# Upsample Minority classes to match Majority (Neutral) count
df_neg_upsampled = resample(df_neg, replace=True, n_samples=len(df_neutral), random_state=42)
df_pos_upsampled = resample(df_pos, replace=True, n_samples=len(df_neutral), random_state=42)

# Combine back into a balanced training set
train_df_balanced = pd.concat([df_neutral, df_neg_upsampled, df_pos_upsampled])
y_train = train_df_balanced['label']
y_val = val_df['label']
y_test = test_df['label']

print(f"Original Train Size: {len(train_df)}")
print(f"Balanced Train Size: {len(train_df_balanced)}")


# 4. Feature Extraction
# Define Feature Functions
def get_manual_features(data_frame):
    # Ensure text column is string
    texts = data_frame['clean_text'].astype(str)
    
    # Feature: Word Count
    word_counts = texts.apply(lambda x: len(x.split()))
    
    # Feature: Character Length
    char_counts = texts.apply(len)
    
    # Scale features (0-1)
    features = pd.DataFrame({'word_count': word_counts, 'char_count': char_counts})
    return features

# A. TF-IDF (Fit on TRAIN only, Transform Val/Test)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df_balanced['clean_text'])
X_val_tfidf = tfidf.transform(val_df['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])

# B. Manual Features
scaler = MinMaxScaler()
# Calculate raw features
X_train_man = get_manual_features(train_df_balanced)
X_val_man = get_manual_features(val_df)
X_test_man = get_manual_features(test_df)

# Fit scaler on TRAIN, transform others
X_train_man = scaler.fit_transform(X_train_man)
X_val_man = scaler.transform(X_val_man)
X_test_man = scaler.transform(X_test_man)


# 5. Combine & Save
X_train = hstack([X_train_tfidf, X_train_man])
X_val = hstack([X_val_tfidf, X_val_man])
X_test = hstack([X_test_tfidf, X_test_man])

with open("data_splits.pkl", "wb") as f:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

print("✅ Feature Extraction & Balancing Complete")