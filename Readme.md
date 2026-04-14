# 🎭 Emotiq — AI Emotion Detector

<p align="center">
  <img src="banner.png" alt="Emotiq Banner" width="100%">
</p>

<p align="center">
  <b>Detect human emotions from text using Machine Learning</b><br>
  Fast • Accurate • Beautiful UI
</p>

---

## 🌐 Live Demo

🚀 **Try it here:**  
👉 https://YOUR-STREAMLIT-APP-URL.streamlit.app  

---

## ✨ Features

🎯 **6 Emotion Classes**  
Sadness • Anger • Love • Surprise • Fear • Joy  

📊 **Confidence Scores**  
Visual probability breakdown for each emotion  

⚡ **Real-Time Prediction**  
Instant results with smooth UI experience  

🧪 **Sample Inputs**  
Test quickly using pre-built examples  

🔍 **Preprocessing Viewer**  
See how text is cleaned before prediction  

🎨 **Modern UI**  
Minimal, editorial design with animations  

---

## 🧠 Model Performance

| Metric        | Value |
|--------------|------|
| Accuracy      | ~86% |
| Model         | Logistic Regression |
| Vectorizer    | TF-IDF (10,000 features) |
| NLP Library   | NLTK |

### 🧹 Preprocessing Steps
- Lowercasing  
- Punctuation removal  
- Stopword removal  

---

## 📸 Preview

<p align="center">
  <img src="preview.png" width="80%">
</p>

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detector.git
cd emotion-detector
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train the Model

Run:

simple.ipynb

This will generate:

model/emotion_model.pkl  
model/tfidf_vectorizer.pkl
4️⃣ Run the App
streamlit run app.py
📁 Project Structure
emotion-detector/
│
├── app.py
├── simple.ipynb
├── model/
│   ├── emotion_model.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
🛠 Tech Stack
Frontend: Streamlit
ML Model: Scikit-learn (Logistic Regression)
Vectorization: TF-IDF
NLP: NLTK
Styling: Custom CSS
🎨 UI Highlights
Editorial typography (Playfair + DM Sans)
Emotion-based color themes
Animated confidence bars
Clean and minimal layout
⚙️ Deployment

Easily deploy using:

Streamlit Cloud
Render
Hugging Face Spaces
💡 Future Improvements
🔊 Voice emotion detection
🌍 Multi-language support
🧠 Deep learning models (BERT / LSTM)
📱 Mobile responsiveness
🤝 Contributing

Contributions are welcome!

fork → clone → branch → commit → push → pull request
