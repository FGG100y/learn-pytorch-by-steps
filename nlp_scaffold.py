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
import numpy as np
import torch
import torch.nn as nn

from gensim import corpora
from gensim.parsing.preprocessing import (
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    preprocess_string,
)

from gensim.utils import simple_preprocess


from datasets import load_dataset, Split
from transformers import (
    #  DataCollatorForLanguageModeling,
    #  BertModel,
    BertTokenizer,
    #  BertForSequenceClassification,
    #  DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    #  AutoModel,
    #  AutoTokenizer,
    #  AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextClassificationPipeline,
)
from transformers.pipelines import SUPPORTED_TASKS


#  from src.model.training_framework import MyTrainingClass


localfolder = "texts"
verbose = 2


# PART-01 create datasets (detail see ./nlp_scaffold_createdata.py)
# get file's path that contain 'sent' in filename in localfolder
def get_file_path(localfolder):
    return [
        os.path.join(localfolder, f)
        for f in os.listdir(localfolder)
        if "sent" in f
    ]


new_fnames = get_file_path(localfolder)
if len(new_fnames) == 0:  # if no such files, create them first
    cmd = "python nlp_scaffold_createdata.py"
    raise FileNotFoundError(f"files not found, run '{cmd}' to create first")


# PART-02 load data by huggingface "datasets"
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
    print("Before train-test-split\n", dataset)


# STEP-03 shuffle the dataset
# shuffle the dataset to get shuffled_dataset,
# use its method 'train_test_split' with test size of 0.2 to get split_dataset,
# whihch contains training and test sets:
shuffled_dataset = dataset.shuffle(seed=42)
split_dataset = shuffled_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

if verbose:
    print("After train-test-split\n", split_dataset)


# PART-03 word tokenization

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

# STEP-02: vocabulary (a list of unique words that appear in the text corpara)
# build our own vocabulary by tokenizing training set
sentences = train_dataset["sentence"]
tokens = [simple_preprocess(sent) for sent in sentences]
dict_tokens = corpora.Dictionary(tokens)  # not regular python dict
if verbose:
    #  print(dict_tokens)
    # some usefull attributes of dict_tokens:
    print(dict_tokens.num_docs)  # how many docs(sentences here)
    print(dict_tokens.num_pos)  # how many tokens(words) over all docs
    #  print(dict_tokens.token2id)  # {unique-word: unique-ID}
    #  print(dict_tokens.cfs)  # collection frequencies #(given token appear)
    #  print(dict_tokens.dfs)  # document frequencies (ID over distinct docs)

# convert a list of tokens into a list of their corresponding indices in vocab
sent_a = "follow the white rabbit"
new_tokens = simple_preprocess(sent_a)
ids = dict_tokens.doc2idx(new_tokens)
#  if verbose:
#      print(new_tokens)
#      print(ids)

# special token: [UNK] for unknown word (words not in our vocabulary)
special_tokens = {"[PAD]": 0, "[UNK]": 1}
dict_tokens.patch_with_special_tokens(special_tokens)


def get_rare_ids(dict_tokens, min_freq):
    return [t[0] for t in dict_tokens.cfs.items() if t[1] < min_freq]


def make_vocab(
    sentences, folder=None, special_tokens=None, vocab_size=None, min_freq=None
):
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
        dictionary.filter_tokens(bad_ids=rare_tokens)
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
    with open(os.path.join(folder, "vocab.txt"), "w") as f:
        for word in words:
            f.write(f"{word}\n")


make_vocab(
    train_dataset["sentence"],
    folder="texts/vocab",
    special_tokens=["[PAD]", "[UNK]"],
    min_freq=2,
)

#  breakpoint()

# use BertTokenizer to create tokenizer based on our own vocabulary,
# NOTE that the pre-trained tokenizer use for real with a pre-trained BERT
# model does not need a vocabulary.
using_local_vocab = False
if using_local_vocab:
    tokenizer = BertTokenizer(
        "texts/vocab/vocab.txt"
    )  # FIXME vocab replacement not needed
    sent_b = "follow the white rabbit neo"
    new_tokens = tokenizer.tokenize(sent_b)
    if verbose:
        print(new_tokens)  # 'neo' should be [UNK]
        breakpoint()

    new_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    # NOTE that tokenizer.encode() will perform two steps above together:
    new_ids_bak = tokenizer.encode(sent_b)
    print(new_ids_bak == new_ids)  # [PAD],[UNK] not in new_ids
    breakpoint()

    # for an enriched output, do call tokenizer() (without specified method)
    token_sent_b = tokenizer(
        sent_b, add_special_tokens=False, return_tensors="pt"
    )
    print(token_sent_b)
    breakpoint()

    # STEP-03 tokenize the dataset
    # (this tokenized_dataset is ready for BERT model input)
    #  Behind these, BERT is actually using vectors to represent the words.
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
    breakpoint()


def contextual_embedding_using_flair_demo(verbose=False):
    # NOTE ELMo needs allennlp==0.9.0 (which may conflict to other pkgs)
    from flair.data import Sentence
    from flair.embeddings import ELMoEmbeddings, TransformerWordEmbeddings
    from flair.embeddings import TransformerDocumentEmbeddings

    watch1 = """
    The Hatter was the first to break the silence. `What day of the
    month is it?' he said, turning to Alice: he had taken his watch out
    of his pocket, and was looking at it uneasily, shaking it every now
    and then, and holding it to his ear.
    """
    watch2 = """
    Alice thought this a very curious thing, and she went nearer to
    watch them, and just as she came up to them she heard one of them
    say, `Look out now, Five! Don't go splashing paint over me like
    that!
    """
    sentences = [watch1, watch2]
    flair_sentences = [Sentence(s) for s in sentences]
    if verbose:
        print(flair_sentences[0])
    # using ELMo model
    elmo = ELMoEmbeddings()
    elmo.embed(flair_sentences)
    token_watch1 = flair_sentences[0].tokens[31]
    token_watch2 = flair_sentences[1].tokens[13]
    similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
    score = similarity(token_watch1.embedding, token_watch2.embedding)
    print("CosineSimilarity of 'watch' in two sentences:", score)
    watch1_elmo_emb = get_embeddings(elmo, watch1)
    print("elmo-embedding of watch1 tokens:\n", watch1_elmo_emb)
    # using BERT model
    bert = TransformerWordEmbeddings("bert-base-uncased", layers="-1")
    embed1 = get_embeddings(bert, watch1)
    embed2 = get_embeddings(bert, watch2)
    bert_watch1 = embed1[31]
    bert_watch2 = embed2[13]
    score = similarity(bert_watch1, bert_watch2)
    print("CosineSimilarity of 'watch' in two sentences:", score)

    # NOTE document embedding
    # generate embeddings for whole documents instead of for single words, thus
    # eliminating the need to average word embeddings:
    documents = [Sentence(watch1), Sentence(watch2)]  # flair_sentences
    bert_doc = TransformerDocumentEmbeddings("bert-base-uncased")
    bert_doc.embed(documents)
    print(documents[0].tokens[31].embedding)  # no individual token's embedding
    print(get_embeddings(bert_doc, watch1))


# word embeddings for all tokens in a sentence, we can simply stack them up:
def get_embeddings(embed_model, sentence):
    """get word or document embeding"""
    from flair.data import Sentence

    sent = Sentence(sentence)
    embed_model.embed(sent)
    if len(sent.embedding):
        return sent.embedding.float()
    else:
        return torch.stack([token.embedding for token in sent.tokens]).float()


#  if verbose:
#      contextual_embedding_using_flair_demo(verbose=True)


# NOTE flair not working (2023-10-12 16:40:31 Thursday)
# PART-04 using pre-trained BERT embeddimgs (document tokenizer)
#  ------ train datasets for pre-trained model ------
#  Before, the features were a sequence of token IDs, which were used to look
#  embeddings up in the embedding layer and return the corresponding bag-of-
#  embeddings (that was a document embedding too, although less sophisticated).
#
#  Now, we’re outsourcing these steps to BERT and getting document embeddings
#  directly from it. It turns out, using a pre-trained BERT model to retrieve
#  document embeddings is a preprocessing step in this setup. Consequently, our
#  model is going to be nothing other than a simple classifier.

#  bert_doc = TransformerDocumentEmbeddings("bert-base-uncased")
#  train_dataset_doc = train_dataset.map(
#      lambda row: {"embeddings": get_embeddings(bert_doc, row["sentence"])}
#  )
#  test_dataset_doc = test_dataset.map(
#      lambda row: {"embeddings": get_embeddings(bert_doc, row["sentence"])}
#  )
#  # format dataset
#  train_dataset_doc.set_format(type="torch", columns=["embeddings", "labels"])
#  test_dataset_doc.set_format(type="torch", columns=["embeddings", "labels"])
#  if verbose:
#      print(train_dataset_doc["embeddings"])
#
#  # model input preparation:
#  train_dataset_doc = TensorDataset(
#      train_dataset_doc["embeddings"].float(),
#      train_dataset_doc["labels"].view(-1, 1).float(),
#  )
#  generator = torch.Generator()
#  train_loader = DataLoader(
#      train_dataset_doc, batch_size=32, shuffle=True, generator=generator
#  )
#  test_dataset_doc = TensorDataset(
#      test_dataset_doc["embeddings"].float(),
#      test_dataset_doc["labels"].view(-1, 1).float(),
#  )
#  test_loader = DataLoader(test_dataset_doc, batch_size=32, shuffle=True)
#
#  # model configuration: (a very simple MLP model)
#  torch.manual_seed(41)
#  model = nn.Sequential(
#      # Classifier
#      nn.Linear(bert_doc.embedding_length, 3),
#      nn.ReLU(),
#      nn.Linear(3, 1),
#  )
#  loss_fn = nn.BCEWithLogitsLoss()
#  optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#  # model training (the MyTrainingClass() one last time)
#  mtc_doc_emb = MyTrainingClass(model, loss_fn, optimizer)
#  mtc_doc_emb.set_loaders(train_loader, test_loader)
#  mtc_doc_emb.train(20)
#  # model performance:
#  fig = mtc_doc_emb.plot_losses()
#  print(MyTrainingClass.loader_apply(test_loader, mtc_doc_emb.correct))
#
#  breakpoint()


#  # PART-05 finetume pre-trained model:
#  bert_model = BertModel.from_pretrained("bert-base-uncased")
#  print(bert_model.config)
#  bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#  print(bert_tokenizer.vocab)


print(SUPPORTED_TASKS['text-classification'])  # alias 'sentiment-analysis'
print(SUPPORTED_TASKS['text-generation'])
breakpoint()


# PART-06 finetume pre-trained model using huggingface Trainer()
def tokenize(row):
    return auto_tokenizer(
        row["sentence"], truncation=True, padding="max_length", max_length=30
    )


bert_cls = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
auto_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize, batched=True)
if verbose:
    print(tokenized_train_dataset[0])
    breakpoint()

# format dataset (use only the first three columns)
tokenized_train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

trainer = Trainer(model=bert_cls, train_dataset=tokenized_train_dataset)
print(trainer.args)
breakpoint()

# customize training args
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=300,
    logging_steps=300,
    gradient_accumulation_steps=8,
)


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


trainer = Trainer(
    model=bert_cls,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

print(trainer.evaluate())
breakpoint()

# save model:
trainer.save_model("models/bert_alice_vs_wizard")
print(os.listdir("models/bert_alice_vs_wizard"))
breakpoint()

# load local model and make predition:
loaded_model = AutoModelForSequenceClassification.from_pretrained(
    "models/bert_alice_vs_wizard"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.to(device)

sentence = "Down the yellow brick rabbit hole"

using_pipeline = False
if using_pipeline:
    # using pipeline:
    device_index = (
        loaded_model.device.index if loaded_model.device.type != "cpu" else -1
    )
    classifier = TextClassificationPipeline(
        model=loaded_model, tokenizer=auto_tokenizer, device=device_index
    )
    # setting proper label for more intuition:
    loaded_model.config.id2label = {0: 'Wizard', 1: 'Alice'}
    print(classifier(['Down the Yellow Brick Rabbit Hole', 'Alice rules!']))
    breakpoint()
else:
    tokens = auto_tokenizer(sentence, return_tensors="pt")
    tokens.to(loaded_model.device)
    loaded_model.eval()
    logits = loaded_model(
        input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
    )
    print(logits.logits.argmax(dim=1))
    breakpoint()


#  # other tasks and underline models:
#  print(SUPPORTED_TASKS['sentiment-analysis'])
#  print(SUPPORTED_TASKS['text-generation'])
breakpoint()
