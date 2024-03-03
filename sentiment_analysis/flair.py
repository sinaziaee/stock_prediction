# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
print(torch.cuda.is_available())
DEVICE = torch.device('cuda')
print(DEVICE)

# %%
# Load dataframe
df = pd.read_csv('../datasets/sentiment_dataset.csv',
                names=['sentiment', 'text'],
                encoding='utf-8', encoding_errors='replace')
# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Load pretrained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize texts and convert to tensors
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

train_labels_encoded = [tokenizer.encode(label, add_special_tokens=False)[0] for label in train_labels]
val_labels_encoded = [tokenizer.encode(label, add_special_tokens=False)[0] for label in val_labels]

label_to_index = {'negative': 0, 'neutral': 1, 'positive': 2}
train_labels_encoded = [label_to_index[label] for label in train_labels]
val_labels_encoded = [label_to_index[label] for label in val_labels]

train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels_encoded)
)

val_dataset = TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels_encoded)
)

# Load pretrained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
# Training parameters
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 6
batch_size = 32

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

train_loss_list = []
val_loss_list = []
val_accuracy_list = []
train_accuracy_list = []
# Fine-tuning loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    temp_train_accuracy_list = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        model = model.to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1
        loss.backward()
        optimizer.step()
        
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions
        temp_train_accuracy_list.append(accuracy)
    avg_train_accuracy = sum(temp_train_accuracy_list) / len(temp_train_accuracy_list)
    avg_loss = total_loss / len(train_dataloader)
    train_loss_list.append(avg_loss)
    train_accuracy_list.append(avg_train_accuracy)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Train Accuracy: {100 * avg_train_accuracy:.2f}')
    # Validation
    model.eval()
    val_loss = 0
    num_batches = 0
    temp_val_accuracy_list = []
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        model = model.to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        val_loss += outputs.loss.item()
        num_batches += 1
        # Calculate accuracy
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions
        temp_val_accuracy_list.append(accuracy)

    avg_val_loss = val_loss / num_batches
    avg_val_accuracy = sum(temp_val_accuracy_list) / len(temp_val_accuracy_list)

    val_loss_list.append(avg_val_loss)
    val_accuracy_list.append(avg_val_accuracy)

    print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {100*avg_val_accuracy:.2f}%')
# Evaluation (optional)
# After training, you can evaluate the model on a separate test set if available
model.save_pretrained("../flair_model")

# %%



