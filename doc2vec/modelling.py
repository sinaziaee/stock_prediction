from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
from collections import Counter
import preprocess
import os
import ast

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 512
BATCH_SIZE = 16
pre_trained_model_ckpt = 'bert-base-uncased'

class DistributedMemory(nn.Module):
    def __init__(self, vec_dim, n_docs, n_words):
        super(DistributedMemory, self).__init__()
        self.paragraph_matrix = nn.Parameter(torch.randn(n_docs, vec_dim))
        self.word_matrix = nn.Parameter(torch.randn(n_words, vec_dim))
        self.outputs = nn.Parameter(torch.zeros(vec_dim, n_words))
    
    def forward(self, doc_ids, context_ids, sample_ids):
                                                                               # first add doc ids to context word ids to make the inputs
        inputs = torch.add(self.paragraph_matrix[doc_ids,:],                   # (batch_size, vec_dim)
                           torch.sum(self.word_matrix[context_ids,:], dim=1))  # (batch_size, 2x context, vec_dim) -> sum to (batch_size, vec_dim)
                                                                               #
                                                                               # select the subset of the output layer for the NCE test
        outputs = self.outputs[:,sample_ids]                                   # (vec_dim, batch_size, n_negative_samples + 1)
                                                                               #
        return torch.bmm(inputs.unsqueeze(dim=1),                              # then multiply with some munging to make the tensor shapes line up 
                         outputs.permute(1, 0, 2)).squeeze()                   # -> (batch_size, n_negative_samples + 1)

class NCEDataset(Dataset):
    def __init__(self, examples):
        self.examples = list(examples)  # just naively evaluate the whole damn thing - suboptimal!
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

class NoiseDistribution:
    def __init__(self, vocab):
        self.probs = np.array([vocab.freqs[w] for w in vocab.words])
        self.probs = np.power(self.probs, 0.75)
        self.probs /= np.sum(self.probs)
        
    def sample(self, n):
        return np.random.choice(a=self.probs.shape[0], size=n, p=self.probs)
    

class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
        
    def forward(self, scores):
        batch_size = scores.shape[0]
        n_negative_samples = scores.shape[1] - 1
        positive = self.log_sigmoid(scores[:,0])
        negatives = torch.sum(self.log_sigmoid(-scores[:, 1:]), dim=1)
        return -torch.sum(positive + negatives) / batch_size

class Vocab:
    def __init__(self, all_tokens, min_count=2):
        self.min_count = min_count
        self.freqs = {t:n for t, n in Counter(all_tokens).items() if n >= min_count}
        self.words = sorted(self.freqs.keys())
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_ckpt, return_dict=False)
        num_parameters = self.bert.num_parameters()
        print(f"The model has {num_parameters} parameters.")
        num_layers = self.bert.config.num_hidden_layers
        print(f"The model has {num_layers} hidden layers.")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        
        output = self.drop(pooled_output)
        return self.out(output)

def example_generator(df, context_size, noise, n_negative_samples, vocab):
    for doc_id, doc in df.iterrows():
        for i in range(context_size, len(doc.clean_tokens) - context_size):
            positive_sample = vocab.word2idx[doc.clean_tokens[i]]
            sample_ids = noise.sample(n_negative_samples).tolist()
            # Fix a wee bug - ensure negative samples don't accidentally include the positive
            sample_ids = [sample_id if sample_id != positive_sample else -1 for sample_id in sample_ids]
            sample_ids.insert(0, positive_sample)                
            context = doc.clean_tokens[i - context_size:i] + doc.clean_tokens[i + 1:i + context_size + 1]
            context_ids = [vocab.word2idx[w] for w in context]
            yield {"doc_ids": torch.tensor(doc_id),  # we use plural here because it will be batched
                   "sample_ids": torch.tensor(sample_ids), 
                   "context_ids": torch.tensor(context_ids)}
            



if __name__ == '__main__':
    train_path = os.path.join('Source', 'Data', 'Train', 'train_stock_news.csv')
    df = preprocess.load_dataset()
    example_df = preprocess.tokenize_text(df)
    print(example_df)
    vocab = Vocab([tok for tokens in example_df.tokens for tok in tokens], min_count=1)
    
    print(f"Dataset comprises {len(df)} documents and {len(vocab.words)} unique words (over the limit of {vocab.min_count} occurrences)")
    
    # vocab = Vocab([tok for tokens in df.tokens for tok in tokens], min_count=1)
    example_df = preprocess.clean_tokens(example_df, vocab)

    # print(f"Dataset comprises {len(df)} documents and {len(vocab.words)} unique words (over the limit of {vocab.min_count} occurrences)")
    print(example_df[:5])
    noise = NoiseDistribution(vocab)
    loss = NegativeSampling()
    
    examples = example_generator(example_df, context_size=5, noise=noise, n_negative_samples=5, vocab=vocab)
    dataset = NCEDataset(examples)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=True)  # TODO bigger batch size when not dummy data
    len(dataloader)
    model = DistributedMemory(vec_dim=50,
                          n_docs=len(example_df),
                          n_words=len(vocab.words))