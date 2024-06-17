from transformers import BertModel, BertTokenizer
import torch

def text_to_bert_embedding(text):
    # Load pre-trained BERT model and tokenizer
    model_path = "..\\uncased_L-12_H-768_A-12"
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        bert_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Take the mean of the last layer's hidden states

    return bert_embedding.numpy()
