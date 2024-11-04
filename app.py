import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from torch import nn

# Define the BengaliBERTModel class for loading
class BengaliBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BengaliBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model with architecture
model_path = "model.pkl"  # Update this to your model's path
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Streamlit app code
st.title("Bengali Sentiment Analysis")
st.write("Enter a sentence in Bengali to predict its sentiment class.")

input_text = st.text_input("Input Text", "")

if st.button("Predict"):
    if input_text:
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(logits, dim=1).item()

        st.write(f"Predicted Class: {predicted_class}")
    else:
        st.write("Please enter some text to predict.")
