"""Subword embeddings

- fastText: Instead of learning word-level vector representations, fastText can
  be considered as the subword-level skip-gram, where each center word is rep-
  resented by the sum of its subword vectors.
  In fastText, for any word 𝑤, denote by 'G𝑤' the union of all its subwords of
  length between 3 and 6 and its special subword. The vocabulary is the union
  of the subwords of all words.
  pro: rare words even out-of-vocab words get better vector representations
  con: larger vocab (size not predefined), and higher computational complexity

- Byte Pair Encoding (BPE): a compression algorithm to allow for
  variable-length subwords in a fixed-size vocabulary. BPE performs a
  statistical analysis of the training dataset to discover common symbols
  within a word, such as consecutive characters of arbitrary length.
  Starting from symbols of length 1, BPE iteratively merges the most frequent
  pair of consecutive symbols to produce new longer symbols. Then such symbols
  can be used as subwords to segment words.

• The fastText model proposes a subword embedding approach. Based on the
skip-gram model in word2vec, it represents a center word as the sum of its
subword vectors.

• BPE performs a statistical analysis of the training dataset to discover
common symbols within a word. As a greedy approach, byte pair encoding
iteratively merges the most frequent pair of consecutive symbols.

• Subword embedding may improve the quality of representations of rare words
and out- of-dictionary words.
"""

import collections


# BPE: ASCIIs and special end-of-word symbol '_' and unknown symbol '[UNK']
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']


# predefined dataset:
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
print(token_freqs)


def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # key of 'pairs' is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)


def merge_symbols(max_freq_pair, token_freqs, symbols):
    """Greedy approach merging"""
    symbols.append("".join(max_freq_pair))  # changing var
    new_token_freqs = {}
    for token, freq in token_freqs.items():
        new_token = token.replace(
                " ".join(max_freq_pair), "".join(max_freq_pair)
        )
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs


#  num_merges = 15  # ValueError: max() empty; line-46
#  num_merges = 12  # gives back the 'raw_token_freqs'
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f"merge #{i + 1}:", max_freq_pair)
print("BPE learned result symbols:", symbols)
print("BPE segment of subwords:", list(token_freqs.keys()))


# NOTE that the result of BPE depends on the dataset being used.
# we can also use the subwords learned from one dataset to segment words of
# another dataset.
def segment_bpe(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start : end] in symbols:
                cur_output.append(token[start : end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(" ".join(cur_output))
    return outputs


tokens = ["tallest_", "fatter_"]
print("Apply to new dataset:", segment_bpe(tokens, symbols))
