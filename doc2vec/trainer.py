from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
from modelling import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.optim import Adam  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        _, predds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(predds == targets).cpu()
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        optimizer.zero_grad()
        
    return correct_predictions / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples): 
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        _, predds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(predds == targets).cpu()
        
        losses.append(loss.item())
        
    return correct_predictions/n_examples, np.mean(losses)

def get_predictions(model, data_loader): 
    model = model.eval()
    news_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            texts = np.array(data_loader.dataset.news)[d['review_text'].tolist()]
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predds = torch.max(outputs, dim=1)
            
            probs = torch.softmax(outputs, dim=1)
            
            news_texts.extend(texts)
            predictions.extend(predds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    return news_texts, predictions, prediction_probs, real_values
    
def train_doc(model, dataloader, loss, epochs=1, lr=1e-3):
    print("Started Training")
    optimizer = Adam(model.parameters(), lr=lr)
    training_losses = []
    
    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        for batch in dataloader:
            
            model.zero_grad()
            logits = model.forward(**batch)
            batch_loss = loss(logits)
            epoch_losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
                
        training_losses.append(np.mean(epoch_losses))
        saving_path = os.path.join('..', 'saved_models')
        os.makedirs(saving_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(saving_path, 'doc_2_vec_model.bin'))
    print("Finished Training")
    print("-------shape:", logits.cpu().detach().numpy().shape)
    
    return training_losses