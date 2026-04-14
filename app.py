import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

# ── Emotion Config ────────────────────────────────────────────────────────────
EMOTIONS = {
    0: {"label": "Sadness",  "emoji": "😢", "color": "#4A90D9"},
    1: {"label": "Anger",    "emoji": "😠", "color": "#E74C3C"},
    2: {"label": "Love",     "emoji": "❤️", "color": "#E91E8C"},
    3: {"label": "Surprise", "emoji": "😲", "color": "#F39C12"},
    4: {"label": "Fear",     "emoji": "😨", "color": "#8E44AD"},
    5: {"label": "Joy",      "emoji": "😄", "color": "#27AE60"},
}

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/emotion_model.pkl"
    vectorizer_path = "model/tfidf_vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ── Text Preprocessing ────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(c for c in text if not c.isdigit())
    text = ''.join(c for c in text if c.isascii())
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; font-size:2.6rem;'>🎭 Emotiq — Feel what words really mean</h1>
    <p style='text-align:center; color:gray; margin-top:-10px;'>
         Paste any text and instantly detect the underlying emotion — powered by Logistic Regression + TF-IDF.
    </p>
    <hr style='margin-bottom:2rem;'>
""", unsafe_allow_html=True)

model, vectorizer = load_model()

if model is None:
    st.warning("⚠️ Model files not found. Please run `save_model.py` first to generate `emotion_model.pkl` and `tfidf_vectorizer.pkl`.", icon="⚠️")
    st.info("**Steps to get started:**\n1. Run `save_model.py` in your project folder\n2. Place `emotion_model.pkl` and `tfidf_vectorizer.pkl` in the same directory as `app.py`\n3. Launch: `streamlit run app.py`")
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "Enter your text below:",
        placeholder="e.g. I feel so happy today! Everything is going great.",
        height=130,
        label_visibility="visible",
    )
with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ── Sample Texts ──────────────────────────────────────────────────────────────
st.markdown("**Try a sample:**")
samples = {
    "😢 Sadness": "I feel so hopeless and lost right now, nothing seems to go right.",
    "😄 Joy":     "I feel strong and good overall, today was absolutely amazing!",
    "😠 Anger":   "I am feeling grouchy and this rude comment made me furious.",
    "❤️ Love":    "I am feeling nostalgic about the fireplace and all those warm memories.",
    "😨 Fear":    "I feel so scared and anxious about what might happen next.",
    "😲 Surprise":"I felt completely shocked and astonished when I heard the news.",
}

scols = st.columns(3)
for i, (label, text) in enumerate(samples.items()):
    if scols[i % 3].button(label, key=f"sample_{i}", use_container_width=True):
        st.session_state["sample_text"] = text
        st.rerun()

if "sample_text" in st.session_state:
    user_input = st.session_state.pop("sample_text")
    st.session_state["last_input"] = user_input

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn and user_input.strip():
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    emo = EMOTIONS[prediction]

    st.markdown("---")
    st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {emo["color"]}22, {emo["color"]}44);
            border-left: 6px solid {emo["color"]};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
        '>
            <h2 style='margin:0; color:{emo["color"]};'>{emo["emoji"]} {emo["label"]}</h2>
            <p style='margin:0.4rem 0 0 0; color:#555;'>Detected emotion with <strong>{probabilities[prediction]*100:.1f}%</strong> confidence</p>
        </div>
    """, unsafe_allow_html=True)

    # Confidence chart for all emotions
    st.markdown("#### Confidence Scores")
    for idx, prob in enumerate(probabilities):
        e = EMOTIONS[idx]
        bar_pct = int(prob * 100)
        st.markdown(f"""
            <div style='margin-bottom:6px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
                    <span>{e["emoji"]} {e["label"]}</span>
                    <span style='color:gray;'>{bar_pct}%</span>
                </div>
                <div style='background:#eee; border-radius:6px; height:10px;'>
                    <div style='
                        background:{e["color"]};
                        width:{bar_pct}%;
                        height:10px;
                        border-radius:6px;
                        transition: width 0.3s;
                    '></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Show preprocessed text
    with st.expander("🔧 See preprocessed text"):
        st.code(processed, language=None)

elif predict_btn and not user_input.strip():
    st.warning("Please enter some text to analyze.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Model: Logistic Regression + TF-IDF &nbsp;|&nbsp; 6 Emotions &nbsp;|&nbsp; ~86% Accuracy"
    "</p>",
    unsafe_allow_html=True,
)