# utils/bert_embedder.py

import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load tokenizer and model once globally
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embedding(text):
    bert_model.eval()
    with torch.no_grad():
        encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        output = bert_model(**encoded)
        embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding.reshape(1, -1)
