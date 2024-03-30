import numpy as np
import pandas as pd
import os
import spacy
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def load_dataset():
    df = pd.read_csv(os.path.join('..', 'datasets', 'stock_news.csv'))
    # create a new dataset with columns news and label
    temp_df = pd.DataFrame(columns=['news', 'label'])
    # concat stock, title and text into news
    temp_df['news'] = df['title'] + ' ' + df['text']
    temp_df.head(2)
    return temp_df

def clean_tokens(df, vocab):
    df['length'] = df.tokens.apply(len)
    df['clean_tokens'] = df.tokens.apply(lambda x: [t for t in x if t in vocab.freqs.keys()])
    df['clean_length'] = df.clean_tokens.apply(len)
    return df

def tokenize_text(df):
    df["tokens"] = df.news.str.lower().str.strip().apply(lambda x: [token.text.strip() for token in nlp(x) if token.text.isalnum()])
    return df