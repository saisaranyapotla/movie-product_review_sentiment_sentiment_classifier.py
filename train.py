"""
Movie Review Sentiment Classifier
Uses the IMDb 50K labeled movie review dataset.
The dataset is downloaded automatically the first time this program runs.
"""
import re
import pickle
import urllib.request
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "IMDB Dataset.csv"
MODEL_PATH = ROOT / "movie_sentiment_model.pkl"
DATA_URL = "https://zenodo.org/records/7928582/files/IMDB%20Dataset.csv?download=1"

def get_dataset():
    if not DATA_PATH.exists():
        print("Downloading IMDb movie review dataset (about 66 MB)...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Dataset downloaded.")
    return pd.read_csv(DATA_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    data = get_dataset().dropna(subset=["review", "sentiment"])
    data["review"] = data["review"].apply(clean_text)
    data["sentiment"] = data["sentiment"].str.lower().str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        data["review"], data["sentiment"], test_size=0.20,
        random_state=42, stratify=data["sentiment"]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB()
    }
    best_model, best_f1 = None, -1
    print("\\nMODEL EVALUATION")
    print("=" * 55)
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=50000)),
            ("classifier", classifier)
        ])
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        print(f"\\n{name}")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))
        if scores["F1-Score"] > best_f1:
            best_f1 = scores["F1-Score"]
            best_model = pipeline
            best_predictions = predictions
            best_name = name

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\\nSaved best model: {MODEL_PATH.name}")

    cm = confusion_matrix(y_test, best_predictions, labels=["negative", "positive"])
    ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    print("\\nMOVIE REVIEW TEST (type exit to stop)")
    while True:
        review = input("Enter a movie review: ").strip()
        if review.lower() == "exit":
            break
        if review:
            print("Predicted sentiment:", best_model.predict([clean_text(review)])[0])

if __name__ == "__main__":
    main()
