# 📰 Factify - Fake News Detection App

**Factify** is an intelligent fake news detector that uses **BERT (DistilBERT)** embeddings along with a machine learning classifier to evaluate whether a news article is **real or fake**, based on its language patterns.

---

### ⚠️ Disclaimer
This tool does **not verify facts or sources**. It analyzes textual **writing style and patterns** to assist in identifying potential misinformation. Always verify news with **credible sources**.

---

### 💡 Features

- ✍️ Paste any news article or paragraph and get instant analysis  
- 🧠 Uses BERT-based semantic understanding for contextual language analysis  
- 📊 Displays confidence scores for predictions  
- 🎨 Clean and modern UI powered by Streamlit  
- 🔒 Fully local execution — no external API calls or data sharing

---

### 🧰 Tech Stack

- **Frontend/UI**: Streamlit (custom themed)
- **Modeling**: HuggingFace Transformers (DistilBERT), Scikit-learn
- **Classifier**: Logistic Regression
- **Data Handling**: Pandas, Torch
- **Persistence**: Joblib (for saving/loading the model)

---

### 📦 Dataset Note

The dataset files (`Fake.csv`, `True.csv`) have been moved into a compressed archive to reduce repository size.  
➡️ You’ll find them inside the `data.zip` file in the root directory.

---

### 👩‍💻 Built By

**Shreeya Soma**
