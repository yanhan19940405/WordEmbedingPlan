import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

use_cuda=torch.cuda.is_available()
#固定随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
#设定超参数
c=3 #词窗数
K=100#负例数
num_epochs=2
max_vocab_size=30000
batch_size=16
learning_rate=0.2
embedding_size=100

def word_tokenize(text):
    return text.split()

with open ("./train.txt",r) as f:
    text=f.read()
print(text)
text=text.split()
vocab=dict(Counter(text).most_common(max_vocab_size-1))
vocab['<unk>']=len(text)-np.sum(list(vocab.values()))

idx_to_word=[word for word in vocab.keys()]
word_to_idx={word:i for i,word in enumerate(idx_to_word)}

word_countx=np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs=word_countx/np.sum(word_countx)
word_freqs=word_freqs**(3./4.)
word_freqs=word_freqs/np.sum(word_freqs)

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, max_vocab_size - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        #返回整个数据集（所有单词）的长度

        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - c, idx)) + list(range(idx + 1, idx + c + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_countx)
dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        #初始化输出和输出embedding
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=2)
        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels: 中心词, [batch_size]
        # pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        # neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        #
        # return: loss, [batch_size]


        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        input_embedding=input_embedding.view(input_embedding.shape[0],1,input_embedding.shape[-1])
        output, (hidden, cell) = self.lstm(input_embedding)

        # hidden: 2 * batch_size * hidden_size
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        input_hidden = hidden.squeeze()
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        pos_output, (pos_hidden, pos_cell) = self.lstm(pos_embedding)
        # hidden: 2 * batch_size * hidden_size
        pos_hidden = torch.cat([pos_hidden[-1], pos_hidden[-2]], dim=1)
        pos_hidden = pos_hidden.squeeze()
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size
        neg_output, (neg_hidden, neg_cell) = self.lstm(neg_embedding)
        # hidden: 2 * batch_size * hidden_size
        neg_hidden = torch.cat([neg_hidden[-1], neg_hidden[-2]], dim=1)
        neg_hidden = neg_hidden.squeeze()
        print(input_hidden.size())
        print(pos_hidden.size())
        print(neg_hidden.size())
        log_pos = torch.matmul(pos_hidden, input_hidden)  # B * (2*C)
        log_neg = torch.matmul(neg_hidden, -input_hidden)  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
model = EmbeddingModel(max_vocab_size, embedding_size,hidden_size=200)
if use_cuda:
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for e in range(num_epochs):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        # TODO
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if use_cuda:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch",e,"iteration",i,loss.item())
        if e==num_epochs-1:
            torch.save(model.state_dict(), "wordemding-model.pth")
print(1)