import os
import dgl
import time
import random
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.sampling import select_topk
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from utils import accuracy, load_data, augmentation
from model import GCN, APPNP, GConv, MLP

'''
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

device = 'cuda:1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='citeseer')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--dropedge', type=float, default=0.0, help='Dropedge rate')
parser.add_argument('--alpha', type=float, default=1.0, help='loss hyperparameter')
parser.add_argument('--temp', type=float, default=0.5, help='sharpen temperature')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()

g, nclass = load_data('cora', split='fix')

g = g.to(device)
feats  = g.ndata['feat']
labels = g.ndata['label']
train  = g.ndata['train_mask']
val    = g.ndata['val_mask']
test   = g.ndata['test_mask']
number = torch.where(train)[0].size()[0]


# main loop
dur = []
log = []
counter = 0

bce_loss = nn.BCELoss(reduction='mean')
mce_loss = nn.NLLLoss(reduction='mean')

seed = 0 
epoch = 500
hidden = 32
dropout = 0.0

# standard prediction
conv = GConv(k=2)
h = conv(g, feats)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    pred = mlp(h)
    loss = mce_loss(pred[train].log(), labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
raw_pred = mlp(h)
acc  = accuracy(raw_pred[val], labels[val])
print('label', acc, 0.0)


# first order neighbor
adj = g.adj(scipy_fmt='coo')
adj = adj - sp.eye(train.size()[0])

index = []
label = []

for i in range(number):
    ids = adj.getrow(i).nonzero()[1].tolist()
    lab = labels[i].item()

    index.extend(ids)
    label.extend([lab for i in range(len(ids))])

index = torch.LongTensor(index).to(device)
label = torch.LongTensor(label).to(device)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    pred = mlp(h)
    loss = mce_loss(pred[index].log(), label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
aug_pred = mlp(h)
aug_acc  = accuracy(aug_pred[val], labels[val])

consis = aug_acc
divers = torch.norm((raw_pred[val] - aug_pred[val]), p='fro').item()
print('first', consis, divers)



# second order neighbor
adj = g.adj(scipy_fmt='coo')
adj2 = np.dot(adj, adj)
adj2[adj2 > 0] = 1
adj = adj2 - adj

index = []
label = []

for i in range(number):
    ids = adj.getrow(i).nonzero()[1].tolist()
    lab = labels[i].item()

    index.extend(ids)
    label.extend([lab for i in range(len(ids))])

index = torch.LongTensor(index).to(device)
label = torch.LongTensor(label).to(device)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    pred = mlp(h)
    loss = mce_loss(pred[index].log(), label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
aug_pred = mlp(h)
aug_acc  = accuracy(aug_pred[val], labels[val])

consis = aug_acc
divers = torch.norm((raw_pred[val] - aug_pred[val]), p='fro').item()
print('second', consis, divers)



# third order neighbor
adj = g.adj(scipy_fmt='coo')
adj2 = np.dot(adj, adj)
adj3 = np.dot(np.dot(adj, adj), adj)
adj2[adj2 > 0] = 1
adj3[adj3 > 0] = 1
adj = adj3 - adj2

index = []
label = []

for i in range(number):
    ids = adj.getrow(i).nonzero()[1].tolist()
    lab = labels[i].item()

    index.extend(ids)
    label.extend([lab for i in range(len(ids))])

index = torch.LongTensor(index).to(device)
label = torch.LongTensor(label).to(device)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    pred = mlp(h)
    loss = mce_loss(pred[index].log(), label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
aug_pred = mlp(h)
aug_acc  = accuracy(aug_pred[val], labels[val])

consis = aug_acc
divers = torch.norm((raw_pred[val] - aug_pred[val]), p='fro').item()
print('third', consis, divers)
