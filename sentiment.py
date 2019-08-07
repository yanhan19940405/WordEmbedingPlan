import torch
from torchtext import data

SEED = 1724

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

import random
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

import torch.nn as nn
import torch.nn.functional as F


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)  # [sent len, batch size, emb dim]
        embedded = embedded.transpose(1,0)
        embedded = embedded.permute(1, 0, 2)  # [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)  # [batch size, embedding_dim]
        return self.fc(pooled)
vocab_size=len(TEXT.vocab)
embedding_dim=100
output_dim=1
pad_idx=TEXT.vocab.stoi(TEXT.pad_token)
model=WordAVGModel(vocab_size=vocab_size, embedding_dim=embedding_dim, output_dim=output_dim, pad_idx=pad_idx)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)

#词向量初始化至预训练词向量，加快模型收敛速度
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[pad_idx] = torch.zeros(embedding_dim)
model.embed.weight.data[UNK_IDX] = torch.zeros(embedding_dim)

#开始训练
optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.train()
    total_len = 0.
    for batch in iterator:
        preds = model(batch.text).squeeze()  # [batch_size]
        #         print(preds.shape, batch.label.shape)
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len


def evaluate(model, iterator, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len


N_EPOCHS = 10
best_valid_acc = 0.
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), "wordavg-model.pth")

    print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)