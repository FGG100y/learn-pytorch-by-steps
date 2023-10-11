""" brief history of Word Embeddings: """
# STEP01 -- Static Word Embeddings:
# 01 One-Hot encoding: each unique token(word) is represented by a vector full
#    of zeros except for one position, which corresponds to the token's index.

# 02 Bag-of-words (BoW): literally a bag of words: it simply sums up the
#    corresponding OHE vectors, completely disregarding any underlying
#    structure or relationships between the words.

# 03 language model (LM): a model that estimates the probability of a token or
#    sequence of tokens. I.e., a LM will predict the tokens more likely to
#    **fill in a blank**.

# 04 N-grams: (n-1) words followed by a blank, an n-gram model.
#    E.g., three words and a blank: a four-gram;

# 05 Continuous Bag-of-words (CBoW), Skip-gram (SG) [all belong to Word2Vec]
#    In these models, the context is given by the surrounding words, both
#    *before* and *after* the blank.
#    E.g., to fill in the following blank:
#       "the small [blank]"
#    for a trigram model, the possibilities are endless. Now, consider the same
#    sentence, this time containing the words that follow the blank:
#       "the small [blank] is barking"
#    much more easy: the blank highly likely is "dog".
#    NOTE the "Continuous" means the vectors are not OHE anymore and have
#    continous values instead.
#    How to find these values that best represent each word? We need to train a
#    model to learn them, and this model is called "Word2Vec".

# 06 GloVe: Global Vectors for Word Representation
#    GloVe combines the skip-gram model with co-occurrence statistics at the
#    global level.

#    NOTE The downside of these static word embeddings is that they are not
#    contextual, which means that we'll always retrieve the same values from
#    the word embeddings, regardless of the actual meaning of the word in
#    context.

# STEP02
# Contextual Word Embeddings: “You shall know a word by the company it keeps”

# 01 ELMo: Embeddings from Language Models (ELMo) is able to understand that
#    words may have different meanings in different contexts. The model is a
#    two-layer bidirectional LSTM encoder using 4,096 dimensions in its cell
#    states and was trained on a really large corpus containing 5.5 billion
#    words. Moreover, ELMo’s representations are *character-based*, so it can
#    easily handle unknown (out-of-vocabulary) words.

# 02 BERT, GPT
#    The general idea, introduced by ELMo, of obtaining contextual word
#    embeddings using a language model still holds true for BERT. The key
#    different part is the **Transformer**.

#    BERT (Bidiertional Encoder Representations from Transformers) is a 
#    transformer-based encoder model.
#    GPT (Generative Pre-trained Transformer) is a transformer-based decoder
#    model.

#    NOTE while BERT was trained to fill in the blanks in the middle of
#    sentences (thus correcting corrupted inputs), GPT was trained to fill in
#    blanks at the end of sentences, effectively predicting the next word in a
#    given sentence.


import os
import json
import errno
import requests
import numpy as np
from pathlib import Path
from copy import deepcopy
from operator import itemgetter
from urllib import parse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

import nltk
from nltk.tokenize import sent_tokenize

import gensim
from gensim import corpora, downloader
from gensim.parsing.preprocessing import (
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    preprocess_string,
)

from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from datasets import load_dataset, Split
from transformers import (
    DataCollatorForLanguageModeling,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline,
    TextClassificationPipeline,
)
from transformers.pipelines import SUPPORTED_TASKS

import nlp_utils as nlpu


# NOTE part-1 create datasets

localfolder = "texts"
verbose = 2


# if lines.cfg file exists, do nothing, else create one
if not os.path.exists(os.path.join(localfolder, "lines.cfg")):
    CFG = """fname,start,end
    alice28-1476.txt,104,3704
    wizoz10-1740.txt,310,5100"""
    # write content of CFG into f'{localfold}/lines.cfg' file
    with open(os.path.join(localfolder, "lines.cfg"), "w") as f:
        f.write(CFG)

# load cfg file and build a dictionary of the form: {fname: (start, end)}
with open(os.path.join(localfolder, "lines.cfg"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split(",") for line in lines[1:]]
    config = {line[0]: (int(line[1]), int(line[2])) for line in lines}

if verbose:
    print("text config:", config)

f1name = Path(parse.urlsplit(nlpu.ALICE_URL).path).name  # "alice28-1476.txt"
f2name = Path(parse.urlsplit(nlpu.WIZARD_URL).path).name  # "wizoz10-1740.txt"


def load_source_text(fname, localfolder="texts", skip_header=False):
    # load alice-text from localfold
    with open(os.path.join(localfolder, fname), "r") as f:
        if skip_header:
            # and get only target lines (readlines 104:3704)
            start, end = config[fname]
        else:
            start, end = None, None
        text = "".join(f.readlines()[slice(start, end, None)])
    return text


try:
    alice_text = load_source_text(fname=f1name, skip_header=True)
    wizoz_text = load_source_text(fname=f2name, skip_header=True)
except FileNotFoundError:
    nlpu.download_text(nlpu.ALICE_URL, localfolder)
    nlpu.download_text(nlpu.WIZARD_URL, localfolder)
    # load texts
    alice_text = load_source_text(fname=f1name, skip_header=True)
    wizoz_text = load_source_text(fname=f2name, skip_header=True)

if verbose == 2:
    print("files in ./texts/\n", os.listdir("./texts/"))

corpus_alice = sent_tokenize(alice_text)
corpus_wizoz = sent_tokenize(wizoz_text)

if verbose:
    print("Num. of sentences in texts:", len(corpus_alice), len(corpus_wizoz))
    if verbose == 2:
        print(corpus_alice[2])
        print(corpus_wizoz[30])


def sentence_tokenize(
    fname,
    quote_char="\\",
    sep_char=",",
    include_header=True,
    include_source=True,
    extensions=("txt"),
    **kwargs,
):
    """cleans line breaks in text file, and save to CSV files"""
    source = load_source_text(fname, skip_header=True)
    contents = source.replace("\n", " ").replace("\r", "")
    corpus = sent_tokenize(contents, **kwargs)

    # builds a CSV file containing tokenized sentences,
    # if include_header is True, and include_source also is true,
    # then using header: ('sentence,source\n'), else ('sentence\n')
    # then write the sentence in corpus in form of:
    # f"{quote_char}{sentence}{quote_char}{sep_char}{fname}"
    base = os.path.splitext(fname)[0]
    new_fname = f"{base}.sent.csv"
    new_fname = os.path.join(localfolder, new_fname)
    with open(new_fname, "w") as f:
        # Header of the file
        if include_header:
            if include_source:
                f.write("sentence,source\n")
            else:
                f.write("sentence\n")
        for sentence in corpus:
            if include_source:
                f.write(
                    f"{quote_char}{sentence}{quote_char}{sep_char}{fname}\n"
                )
            else:
                f.write(f"{quote_char}{sentence}{quote_char}\n")

    return new_fname


# write a function to get file's path that contain 'sent' in filename in localfolder
def get_file_path(localfolder):
    return [
        os.path.join(localfolder, f)
        for f in os.listdir(localfolder)
        if "sent" in f
    ]


new_fnames = get_file_path(localfolder)
if len(new_fnames) == 0:
    new_fnames = []
    for fname in (f1name, f2name):
        new_fname = sentence_tokenize(fname=fname)
        new_fnames.append(new_fname)
if verbose:
    print(new_fnames)


# NOTE part-2 huggingface dataset
# STEP-01 loading a dataset (using load_dataset function)

# loading a dataset using huggingface dataset's load_dataset function
# with parameter data_files=new_fnames, quotechar="\\", split=TRAIN
dataset = load_dataset(
    path="csv", data_files=new_fnames, quotechar="\\", split=Split.TRAIN
)

if verbose:
    # print some attributes of dataset, such as features, columns, shape, etc.
    print(dataset.features)
    print(dataset.column_names)
    print(dataset.shape)
    print(dataset.unique("source"))


# STEP-02 data prepraration


# use map() to create new columns by using function that return dict with new
# columns as key
def is_alice_label(row):
    is_alice = int(row["source"] == "alice28-1476.txt")
    return {"labels": is_alice}


dataset = dataset.map(is_alice_label)
if verbose:
    print(dataset[2])


# shuffle the dataset to get shuffled_dataset,
# then its method 'train_test_split' with test size of 0.2 to get split_dataset,
# whihch contains training and test sets:
shuffled_dataset = dataset.shuffle(seed=42)
split_dataset = shuffled_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

if verbose:
    print(split_dataset)


# NOTE part-03 word tokenization

# STEP-01: use Gensim's preprocess_string() DEMO
# and use only the first four filters for preprocess_string():
if verbose:
    filters = [
        lambda x: x.lower(),
        strip_tags,
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_numeric,
    ]
    sentence = "I'm following the white rabbit"
    print(preprocess_string(sentence, filters=filters))

# STEP-02: vocabulary (a list of unique wordds that appear in the text corpara)
# build our own vocabulary by tokenizing training set
setences = train_dataset["sentence"]
tokens = [simple_preprocess(sent) for sent in sentences]
if verbose:
    print(tokens[0])

dict_tokens = corpora.Dictionary(tokens)  # not regular python dict
if verbose:
    print(dict_tokens)
    # some usefull attributes of dict_tokens:
    print(dict_tokens.num_docs)  # how many docs(sentences here)
    print(dict_tokens.num_pos)  # how many tokens(words) over all docs
    print(dict_tokens.token2id)  # {unique-word: unique-ID}
    print(dict_tokens.cfs)  # collection frequencies #(given token appear)
    print(dict_tokens.dfs)  # document frequencies (ID over distinct docs)

# convert a list of tokens into a list of their corresponding indices in vocab
sent_a = "follow the white rabbit"
new_tokens = simple_preprocess(sent_a)
ids = dict_tokens.doc2idx(new_tokens)
if verbose:
    print(new_tokens)
    print(ids)

# special token: [UNK] for unknown word (words not in our vocabulary)
special_tokens = {'[PAD]': 0, '[UNK]': 1}
dict_tokens.patch_with_special_tokens(special_tokens)


def get_rare_ids(dict_tokens, min_freq):
    return [t[0] for t in dict_tokens.cfs.items() if t[1] < min_freq]


def make_vocab(sentences, folder=None, special_tokens=None, vocab_size=None,
               min_freq=None):
    if folder is not None:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # tokenizes the sentences and creates a Dictionary
    tokens = [simple_preprocess(sent) for sent in sentences]
    dictionary = corpora.Dictionary(tokens)  # not regular python dict
    if vocab_size is not None:
        dictionary.filter_extremes(keep_n=vocab_size)
    if min_freq is not None:
        rare_tokens = get_rare_ids(dictionary, min_freq)
        dictionary.filter_extremes(bad_ids=rare_tokens)
    # gets the whole list of tokens and frequencies
    items = dictionary.cfs.items()
    words = [dictionary[t[0]] for t in sorted(items, key=lambda t: -t[1])]
    # prepends special tokens, if any
    if special_tokens is not None:
        to_add = []
        for special_token in special_tokens:
            if special_token not in words:
                to_add.append(special_token)
        words = to_add + words

    # write words to 'vocab.txt' into folder
    with open(os.path.join(folder, 'vocab.txt'), 'w') as f:
        for word in words:
            f.write(f'{word}\n')


make_vocab(train_dataset["sentence"], folder="texts/vocab",
           special_tokens=["[PAD]", "[UNK]"], min_freq=2)

# use BertTokenizer to create tokenizer based on our own vocabulary,
# NOTE that the pre-trained tokenizer use for real with a pre-trained BERT
# model does not need a vocabulary.
tokenizer = BertTokenizer("texts/vocab.txt")
sent_b = "follow the white rabbit neo"
new_tokens = tokenizer.tokenize(sent_b)
if verbose:
    print(new_tokens)  # 'neo' should be [UNK]

new_ids = tokenizer.convert_tokens_to_ids(new_tokens)
# NOTE that tokenizer.encode() will perform two steps above together:
new_ids_bak = tokenizer.encode(sent_b)
print(new_ids_bak == new_ids)

# for an enriched output, do call tokenizer() (without specified method)
token_sent_b = tokenizer(sent_b, add_special_tokens=False, return_tensors="pt")
print(token_sent_b)

# STEP-03 tokenize the dataset
# (this tokenized_dataset is ready for BERT model input)
#  Behind the curtain, BERT is actually using vectors to represent the words.
#  The token IDs we’ll be sending it are simply the indices of an enormous
#  lookup table. That lookup table has a very nice name: Word Embeddings.
tokenized_dataset = tokenizer(
        dataset["sentence"],
        padding=True,
        return_tensors="pt",
        max_length=50,
        truncation=True,
        )
print(tokenized_dataset)
