# app.py

import streamlit as st
import numpy as np
import torch
from joblib import load
from transformers import DistilBertTokenizer, DistilBertModel

# Load BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()

# Load the trained model
model = load("utils/factify_model.pkl")

# Streamlit Page Config and Theme
st.set_page_config(page_title="Factify - Fake News Detector", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stTextArea textarea {
        border: 2px solid #ccc;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🧠 Factify")
st.sidebar.write("A BERT-powered fake news detector app.")
st.sidebar.markdown("---")
st.sidebar.info("💡 Enter any news text in the box and click 'Analyze' to check if it's Real or Fake.")
st.sidebar.markdown("⚠️ **Disclaimer:** This app uses machine learning to estimate news authenticity based on writing style. It does not verify factual accuracy. Use responsibly.")
st.sidebar.markdown("Built by **Shreeya Soma**")

# Main Title
st.markdown("## 📰 Factify - Real or Fake?")
st.write("Paste any news article or paragraph below and hit 'Analyze'!")
st.markdown("**Factify** uses advanced BERT embeddings + ML to classify the article as **Fake** or **Real**.")

# Text Input
news_text = st.text_area("✍️ Paste the news article text here:")

# BERT Embedding Function
def get_bert_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        output = bert_model(**encoded)
        embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding.reshape(1, -1)

# Button to Analyze
if st.button("🔍 Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                embedding = get_bert_embedding(news_text)
                prediction = model.predict(embedding)[0]
                proba = model.predict_proba(embedding)[0]

                st.success("✅ Analysis Complete!")
                if prediction == 1:
                    st.markdown(f"### 🟢 This news is **REAL** with {proba[1]*100:.2f}% confidence.")
                else:
                    st.markdown(f"### 🔴 This news is **FAKE** with {proba[0]*100:.2f}% confidence.")

                # Inline disclaimer
                st.info("⚠️ *This result is based on linguistic patterns, not verified facts. Always consult official sources for confirmation.*")

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
