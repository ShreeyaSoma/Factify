# utils/model.py

import joblib
from utils.bert_embedder import get_embedding

# Load trained classifier
model = joblib.load("utils/factify_model.pkl")

def predict_news(news_text):
    embedding = get_embedding(news_text)
    prediction = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][prediction]
    return prediction, round(prob * 100, 2)
