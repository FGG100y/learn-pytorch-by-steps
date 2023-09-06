"""
Variable-Length Sequences handling DEMO
1. padding: fill(0) to make sequences equal length
2. packing: works like a concatenation of sequences while keep track of lengths
so that it knows the indices corresponding to the start of each sequence.
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils

from src.data_generation.square_sequences import generate_sequences

points, directions = generate_sequences(n=128, seed=42)

x0 = points[0]  # 4 data points
x1 = points[1][2:]  # 2 data points
x2 = points[2][1:]  # 3 data points

all_seqs = [x0, x1, x2]

rnn = nn.RNN(2, 2, batch_first=True)

#  handling = "packing"
handling = None
try:
    seq_tensors = torch.as_tensor(all_seqs)
    output, hidden = rnn(seq_tensors)
    print(output[:, -1])
except ValueError as e:
    print(e)
    seq_tensors = [torch.as_tensor(seq).float() for seq in all_seqs]
    if handling == "padding":
        print("Handling above error by padding")
        padded = rnn_utils.pad_sequence(seq_tensors, batch_first=True)
        # NOTE that padded points do modify the hidden states
        output_padded, hidden_padded = rnn(padded)
        # permuted, batch-first
        print("final state:\n", hidden_padded.permute(1, 0, 2))
    elif handling == "packing":
        print("Handling above error by packing")
        # NOTE that enfore_sorted=True only necessary for doing ONNX format
        packed = rnn_utils.pack_sequence(seq_tensors, enforce_sorted=False)
        # NOTE also that if the input is packed,
        # the output tensor is packed too, but the hidden state is not.
        output_packed, hidden_packed = rnn(packed)
        # unpacking output into regular, yet padded, output
        output_unpacked, seq_sizes = rnn_utils.pad_packed_sequence(
            output_packed, batch_first=True
        )
        # to get the last output
        # NOTE that the output_unpacked[:, -1] will do no good here,
        # it gives the padding zeros back
        print(output_unpacked[:, -1])
        # the right way to get the last output for each packed sequence:
        seq_idx = torch.arange(seq_sizes.size(0))
        print(output_unpacked[seq_idx, seq_sizes - 1])
    else:  # mix both
        len_seqs = [len(seq) for seq in all_seqs]
        padded = rnn_utils.pad_sequence(seq_tensors, batch_first=True)
        packed = rnn_utils.pack_padded_sequence(
            padded,
            len_seqs,
            enforce_sorted=False,
            batch_first=True,
        )
        output_packed, hidden_packed = rnn(packed)
        print(output_unpacked[:, -1])
        seq_idx = torch.arange(seq_sizes.size(0))
        print(output_unpacked[seq_idx, seq_sizes - 1])
