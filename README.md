# 📰 Factify - Fake News Detection App

**Factify** is an intelligent fake news detector that uses **BERT (DistilBERT)** embeddings along with a machine learning classifier to evaluate whether a news article is **real** or **fake**, based on its **language patterns**.

> ⚠️ *Disclaimer:* This tool does **not verify facts or sources**. It analyzes textual style and patterns to assist in identifying potential misinformation. Always verify news with credible sources.

---

## 💡 Features

- ✍️ Paste any news article or paragraph and get instant analysis
- 🧠 Uses **BERT-based semantic understanding** for language context
- 📊 Displays confidence scores for predictions
- 🎨 Clean and modern UI powered by **Streamlit**
- 🔒 Locally deployed – no data is sent outside

---

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit (custom themed)
- **Modeling**: HuggingFace Transformers (`DistilBERT`), Scikit-learn
- **Classifier**: Logistic Regression
- **Data Handling**: Pandas, Torch
- **Persistence**: Joblib (for saving/loading model)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/Factify.git
cd Factify
