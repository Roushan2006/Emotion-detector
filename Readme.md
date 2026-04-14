# 🎭 Emotiq — Emotion Detector

A sleek, white-themed web app that detects emotions in text using Machine Learning.

## ✨ Features

- **6 Emotion Classes** — Sadness, Anger, Love, Surprise, Fear, Joy
- **Confidence Scores** — Visual breakdown of all emotion probabilities
- **Sample Texts** — One-click sample inputs for each emotion
- **Preprocessing Viewer** — Inspect the cleaned text before vectorization
- **~86% Accuracy** — Logistic Regression + TF-IDF pipeline

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detector.git
cd emotion-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and save the model

simple.ipynb

This generates `model/emotion_model.pkl` and `model/tfidf_vectorizer.pkl`.

### 4. Run the app

```bash
streamlit run app.py
```

## 📁 Project Structure

```
emotion-detector/
├── app.py                  # Streamlit UI
├── simple.ipynb           # Model training & export
├── model/
│   ├── emotion_model.pkl       # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── requirements.txt
└── README.md
```

## 🛠 Tech Stack

| Layer       | Technology              |
|-------------|-------------------------|
| UI          | Streamlit               |
| ML Model    | Scikit-learn (Logistic Regression) |
| Vectorizer  | TF-IDF                  |
| NLP Preprocessing | NLTK           |
| Fonts       | Playfair Display, DM Sans, DM Mono |

## 📊 Model Details

- **Algorithm**: Logistic Regression (multi-class)
- **Features**: TF-IDF with up to 10,000 features
- **Preprocessing**: Lowercasing, punctuation removal, stopword removal
- **Dataset**: Emotion-labeled text dataset (6 classes)

## 🎨 UI Design

Dark editorial aesthetic with:
- Playfair Display serif headings
- DM Sans / DM Mono body text
- Per-emotion color theming on result cards
- Animated confidence bars
- Fully custom Streamlit CSS overrides

