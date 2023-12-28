"""word2vec: skip-gram & negative sampling

Construct pipeline for Penn Tree Bank (PTB) data.
"""
import collections
import math
import os
import random

import joblib
import torch
from d2l import torch as d2l


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset (if needed) then load it into memory"""
    num_workers = d2l.get_dataloader_workers()
    try:
        sentences = joblib.load("./texts/ptbdata.pkl")
    except FileNotFoundError:
        d2l.DATA_HUB["ptb"] = (  # dictionary
            d2l.DATA_URL + "ptb.zip",  # from amazonaws.com
            "319d85e578af0cdc590547f26231e4e31cdf1e42",
        )
        sentences = read_ptb()
        joblib.dump(sentences, "./texts/ptbdata.pkl")
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size
    )
    all_negatives = get_negative(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (
                self.centers[index],
                self.contexts[index],
                self.negatives[index],
            )

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=batchify,
        num_workers=num_workers,
    )
    return data_iter, vocab


# PART-01: Penn Tree Bank (PTB) data
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract("ptb")
    # read the training set
    with open(os.path.join(data_dir, "ptb.train.txt")) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split("\n")]


# PART-02: subsampling (to the high-frequency words, for speed)
# I.e., each indexed word w_i in the dataset will be discarded with prob.:
#       P(w_i) = max(1 - sqrt(t / f(w_i)), 0)
# where f(w_i) is the ratio of the NO. of words w_i to total NO. of words in
# dataset, and the constant t is a hyper-param (0.0001, in experiment).
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # exclude unknown tokens '<unk>'
    sentences = [
        [token for token in line if vocab[token] != vocab.unk]
        for line in sentences
    ]
    counter = collections.Counter(
        [token for line in sentences for token in line]
    )
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return random.uniform(0, 1) < math.sqrt(
            1e-4 / counter[token] * num_tokens
        )

    return (
        [[token for token in line if keep(token)] for line in sentences],
        counter,
    )


def compare_counts(token):
    return (
        f'# of "{token}": '
        f"before={sum([l.count(token) for l in sentences])}, "
        f"after={sum([l.count(token) for l in subsampled])}"
    )


# PART-03: extracting center words and context words
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram"""
    centers, contexts = [], []
    for line in corpus:
        # each sentence needs to have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # context window centered at 'i'
            window_size = random.randint(1, max_window_size)
            indices = list(
                range(
                    max(0, i - window_size),
                    min(len(line), i + 1 + window_size),
                )
            )
            # exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


# PART-04: negative sampling for approximate training
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights"""

    def __init__(self, sampling_weights):
        # exclude index 0
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # check 'k' random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000
            )
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negative(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling"""
    # sampling weights for words with indices 1, 2, ... (index 0 is unk token)
    sampling_weights = [
        counter[vocab.to_tokens(i)] ** 0.75  # (word2vec paper)
        for i in range(1, len(vocab))
    ]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


# PART-05: loading training examples in minibatches
# After all the center words together with their context words and sampled
# noise words are extracted, they will be transformed into minibatches of
# examples that can be iteratively loaded during training.

# issues:
# 1) varying context window size lead to varies input 'i' <- padding
# 2) to distinguish between positive and negative examples


def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (
        torch.tensor(centers).reshape((-1, 1)),
        torch.tensor(contexts_negatives),
        torch.tensor(masks),
        torch.tensor(labels),
    )


if __name__ == "__main__":
    verbose = 1

    try:
        sentences = joblib.load("./texts/ptbdata.pkl")
    except FileNotFoundError:
        d2l.DATA_HUB["ptb"] = (  # dictionary
            d2l.DATA_URL + "ptb.zip",  # from amazonaws.com
            "319d85e578af0cdc590547f26231e4e31cdf1e42",
        )
        sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    if verbose:
        print(f"sentences: {len(sentences)}")
        print(f"vocab size: {len(vocab)}")

    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    if verbose:
        print(compare_counts("the"))
        print(compare_counts("join"))
        print(corpus[:3])
        #  breakpoint()

    if verbose == 2:
        # test with a tiny dataset
        tiny_dataset = [list(range(7)), list(range(7, 10))]
        print("dataset", tiny_dataset)
        for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
            print("center", center, "has contexts:", context)

    all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    if verbose:
        print(f"# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}")  # noqa

    all_negatives = get_negative(all_contexts, vocab, counter, 5)

    if verbose == 1:
        x_1 = (1, [2, 2], [3, 3, 3, 3])
        x_2 = (1, [2, 2, 2], [3, 3])
        batch = batchify((x_1, x_2))

        names = ["centers", "contexts_negatives", "masks", "labels"]
        for name, data in zip(names, batch):
            print(name, "=", data)
