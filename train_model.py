# train_model.py

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# Load BERT model + tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load data
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df['label'] = 0
true_df['label'] = 1

# Combine and sample
df = pd.concat([fake_df, true_df]).sample(n=2500, random_state=42).reset_index(drop=True)

# Function to convert text to BERT embeddings
def get_bert_embeddings(text_list):
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list, desc="Generating BERT embeddings"):
            encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            output = bert_model(**encoded)
            embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

# Prepare data
texts = df['title'] + " " + df['text']
labels = df['label']

X = get_bert_embeddings(texts)
y = labels.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy on test set: {acc:.2f}")

# Save model
joblib.dump(model, "utils/factify_model.pkl")
print("✅ Model saved as 'utils/factify_model.pkl'")
