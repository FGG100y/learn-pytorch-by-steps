"""
Variable-Length Sequences padding DEMO
"""
import torch
from torch.nn.utils import rnn as rnn_utils

from src.data_generation.square_sequences import generate_sequences


points, directions = generate_sequences(n=128, seed=42)

x0 = points[0]      # 4 data points
x1 = points[1][2:]  # 2 data points
x2 = points[2][1:]  # 3 data points

try:
    all_seqs = [x0, x1, x2]
    seq_tensors = torch.as_tensor(all_seqs)
except ValueError as e:
    print(e)
    seq_tensors = [torch.as_tensor(seq).float() for seq in all_seqs]
    padded = rnn_utils.pad_sequence(seq_tensors, batch_first=True)
