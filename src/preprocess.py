# preprocess.py
import re
import string
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Stemmer
stemmer = ISRIStemmer()
# Get Arabic stopwords
stop_words = set(stopwords.words('arabic'))


# Arabic Character Normalization
def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'گ', 'ك', text)
    return text


# Text Cleaning Pipeline
def clean_text(text):
    text = str(text).strip().lower()
    text = normalize_arabic(text)

    # Remove Diacritics (Tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # Process Hashtags (keep content, remove #)
    # The PDF suggests keeping them or processing them; we'll treat them as words
    text = re.sub(r"#([أ-يA-Za-z0-9_]+)", r"\1", text)

    # Remove URLs and Mentions
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Convert Emojis to Special Tokens (Feature Engineering)
    text = re.sub(r'[🤬😡😢😭😞😔💔😠👎]', ' emoji_neg ', text)
    text = re.sub(r'[😊😂😍❤️😁🙂💙💜👍]', ' emoji_pos ', text)

    # Remove Repeated Characters (e.g., هههههه -> هه)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove Punctuation
    punct = string.punctuation + "؟،؛«»"
    text = text.translate(str.maketrans("", "", punct))

    # Remove Non-Arabic/English/Number characters
    text = re.sub(r"[^ء-ي0-9a-zA-Z\s_]", " ", text)

    # Tokenization, Stop Word Removal, and Stemming
    words = text.split()
    cleaned_words = []
    for w in words:
        if w not in stop_words and len(w) > 1:
            # Apply Stemming
            stemmed_w = stemmer.stem(w)
            cleaned_words.append(stemmed_w)

    return " ".join(cleaned_words)


# Load and Process Data
def load_and_clean(path="data.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8")
        
        # MERGE LABELS: OBJ -> NEUTRAL [Requirement]
        df["label"] = df["label"].replace({"OBJ": "NEUTRAL"})
        
        # Clean text
        df["clean_text"] = df["text"].apply(clean_text)
        
        # Drop empty rows after cleaning
        df = df[df["clean_text"].str.strip().astype(bool)]
        
        df.to_csv("clean_data.csv", index=False)
        print("✔️ Preprocessing Complete. Saved to 'clean_data.csv'")
        print("Unique Labels:", df["label"].unique())
        return df
        
    except FileNotFoundError:
        print("❌ Error: data.csv not found. Run txt_to_csv.py first.")

if __name__ == "__main__":
    load_and_clean()