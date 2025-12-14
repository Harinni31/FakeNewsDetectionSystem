# train_model.py

import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords once
nltk.download('stopwords')

# Load datasets
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")


# Add labels
true_news["label"] = 0
fake_news["label"] = 1

# Combine datasets
news_dataset = pd.concat([true_news, fake_news], axis=0)

# Clean data
news_dataset = news_dataset.fillna("")
news_dataset["content"] = news_dataset["title"] + " " + news_dataset["text"]

# Features and labels
X = news_dataset["content"]
Y = news_dataset["label"]

# Initialize stemmer and stopwords ONCE
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Stemming function (optimized)
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

print("🔄 Preprocessing text...")
X = X.apply(stemming)

# TF-IDF Vectorization
print("🔄 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train model
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print("✅ Training Accuracy:", train_acc)
print("✅ Testing Accuracy:", test_acc)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 Model and vectorizer saved successfully!")
