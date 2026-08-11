# Arabic Sentiment Classification Using NLP and Machine Learning

A machine learning project for classifying Arabic social media text into three sentiment categories: **Positive, Negative, and Neutral**.

The project focuses on Arabic text preprocessing, feature engineering, class imbalance handling, and comparative evaluation of multiple machine learning classifiers.

## Overview

Arabic sentiment analysis presents several challenges due to dialects, spelling variations, rich morphology, emojis, informal writing, and noisy social media text.

This project implements a complete sentiment classification pipeline including:

- Arabic text normalization
- URL, mention, punctuation, and noise removal
- Emoji-based sentiment handling
- Arabic stop-word removal
- ISRI Arabic stemming
- TF-IDF feature extraction
- Manual text features
- Class balancing through upsampling
- Train, validation, and test dataset splitting
- Classification using multiple machine learning models

## Models

Three classification models were implemented and compared:

- **Multinomial Naive Bayes**
- **Random Forest**
- **Multilayer Perceptron (MLP) Neural Network**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## Feature Engineering

The project uses a combination of:

- TF-IDF features
- Unigrams and bigrams
- Word count
- Character count
- Normalized manual features

The TF-IDF representation uses up to **5,000 features**.

## Dataset Processing

The original dataset is stored in:

`Tweets.txt`

It is converted into CSV format using:

`txt_to_csv.py`

The preprocessing stage produces:

`clean_data.csv`

The processed data is divided into:

- 60% Training
- 20% Validation
- 20% Testing

Class balancing is performed on the training set by upsampling the minority sentiment classes.

## Project Structure

```text
Arabic-Sentiment-Analysis/
│
├── README.md
├── AI-Project2-report.pdf
│
├── src/
│   ├── main.py
│   ├── txt_to_csv.py
│   ├── preprocess.py
│   ├── features_balancing.py
│   ├── features_without_balancing.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── neural_network.py
│
└── data/
    ├── Tweets.txt
    └── data.csv
