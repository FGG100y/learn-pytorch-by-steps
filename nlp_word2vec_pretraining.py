"""Pretrain word2vec using negative sampling on the PTB dataset.

Implement the skip-gram model by using embedding layers and batch matrix
multiplications.
"""
import math

import torch
from d2l import torch as d2l
from torch import nn

import nlp_word2vec_ptbdata as ptbdata

# load the ptb dataset
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = ptbdata.load_data_ptb(
    batch_size, max_window_size, num_noise_words
)
names = ["centers", "contexts_negatives", "masks", "labels"]
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, "shape:", data.shape)
    break  # print only the first batch


# embedding layer
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)


# skip gram model: forward propagation
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    # transformed from the token indices into vectors via the embedding layer:
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    # then returns an output of shape (batch size, 1, max_len). Each element in
    # the output is the dot product of a center word vector and a context or
    # noise word vector:
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred  # shape of (batch size, 1, max_len)


# training: Binary Cross-Entropy Loss
class SigmoidBCELoss(nn.Module):
    # BCE Loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        return out.mean(dim=1)


bce_loss = SigmoidBCELoss()


# init model parameters
embed_size = 100  # word vector dimension
net = nn.Sequential(
    # two embeding layers for all the words in vocabulary:
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
)


# defining the training loop:
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs]
    )

    # sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch
            ]
            pred = skip_gram(center, context_negative, net[0], net[1])
            loss = (
                bce_loss(
                    pred.reshape(label.shape).float(), label.float(), mask
                )
                / mask.sum(axis=1)
                * mask.shape[1]
            )
            loss.sum().backward()
            optimizer.step()
            metric.add(loss.sum(), loss.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches, (metric[0] / metric[1])
                )
        print(
            f"loss {metric[0] / metric[1]:.3f} "
            f"{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}"
        )


lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)


# apply word embeddings:
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # compute the cosine similarity. add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(
        torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9
    )
    top_k = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype("int32")
    for i in top_k[1:]:  # remove input words
        print(f"cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}")


get_similar_tokens("chip", 3, net[0])
