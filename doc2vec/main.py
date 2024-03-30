# from dataloader import *
# from modeling import *
from trainer import *
# from utils.result import *
import os
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import preprocess
from modelling import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import altair as alt
import pandas as pd
from sklearn.decomposition import PCA

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)

os.makedirs('results', exist_ok=True)

def visualize_loss(training_losses):
    df_loss = pd.DataFrame(enumerate(training_losses), columns=["epoch", "training_loss"])
    chart = alt.Chart(df_loss).mark_bar().encode(alt.X("epoch"), alt.Y("training_loss", scale=alt.Scale(type="log")))
    chart.save('results/training_loss_for_doc2vec.html')
def pca_2d(paragraph_matrix):
    pca = PCA(n_components=2)
    reduced_dims = pca.fit_transform(paragraph_matrix)
    print(f"2-component PCA, explains {sum(pca.explained_variance_):.2f}% of variance")
    df = pd.DataFrame(reduced_dims, columns=["x", "y"])
    return df

def show_confusion_matrix(confusion_matrix):
    sns.set()
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted Sentiment')
    
    # Save the figure
    plt.savefig('results/heatmap.png', dpi=300, bbox_inches='tight')

    plt.close()  # Close the figure to free memory, especially important in scripts
def train_doc2vec(n_epochs, vec_dim=50):
    print("Preprocessing data...")
    df = preprocess.load_dataset()
    print("Tokenizing text...")
    example_df = preprocess.tokenize_text(df)
    print("Cleaning tokens...")
    vocab = Vocab([tok for tokens in example_df.tokens for tok in tokens], min_count=2)
    example_df = preprocess.clean_tokens(example_df, vocab)
    noise = NoiseDistribution(vocab)
    loss = NegativeSampling()
    examples = example_generator(example_df, context_size=5, noise=noise, n_negative_samples=5, vocab=vocab)
    dataset = NCEDataset(examples)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=True)  # TODO bigger batch size when not dummy data
    model = DistributedMemory(vec_dim=vec_dim,
                        n_docs=len(example_df),
                        n_words=len(vocab.words))
    model = model.to(device)
    loss = loss.to(device)
    train_loss = train_doc(model, dataloader, loss, epochs=n_epochs)
    print("TRAINING LOSS:", np.mean(train_loss))
    visualize_loss(training_losses=train_loss)
    example_2d = pca_2d(model.paragraph_matrix.data.detach().cpu())
    chart = alt.Chart(example_2d).mark_point().encode(x="x", y="y")
    chart.save('results/pca_result_for_doc2vec.html')


# print("Input data should be tokenized using the BERT tokenizer.")
# print("Each input should be transformed into input IDs, attention masks, and, for certain tasks, token type IDs.")
# example_text = "Example text input for BERT."
# encoded_input = tokenizer(example_text, return_tensors='pt')
# print("Example of tokenized input:", encoded_input)

train_doc2vec(n_epochs=10, vec_dim=50)

