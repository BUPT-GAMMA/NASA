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

g, nclass = load_data('pubmed', split='fix')
augmentor = augmentation(g, g.ndata['feat'])
augmentor.init(diff='order')

g = g.to(device)
feats  = g.ndata['feat']
labels = g.ndata['label']
train  = g.ndata['train_mask']
val    = g.ndata['val_mask']
test   = g.ndata['test_mask']


# main loop
dur = []
log = []
counter = 0

bce_loss = nn.BCELoss(reduction='mean')
mce_loss = nn.NLLLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')
kld_loss = nn.KLDivLoss(reduction='batchmean')


# cora p=0.5 d=0.3 seed=864
# cite p=0.5 d=0.5 seed=864


seed = 864 
print(seed)
epoch = 500
hidden = 32
dropout = 0.5

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
pred = mlp(h)
acc  = accuracy(pred[val], labels[val])
print('train', acc, 0.0)


# label propagation
def label_prop_loss(edges):
    # If A is a labeled node, B is A's neighbor without label
    # Reduce function follows the aggregation pricinple, i.e., only A <- B will be aggregated by A
    # So, the loss should use dst as label and src as prediction
    label = edges.dst['label']
    pred = edges.src['pred']
    loss = -pred.log()[torch.tensor([i for i in range(label.size()[0])]), label]
    return {'loss': loss}

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    aug_pred = mlp(h)
    with g.local_scope():
        g.ndata['pred'] = aug_pred

        g.update_all(fn.copy_u('pred', '_'), fn.mean('_', 'pred'))
        aug_pred = g.ndata.pop('pred')
        loss = mce_loss(aug_pred.log()[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
aug_pred = mlp(h)
aug_acc  = accuracy(aug_pred[val], labels[val])

consis = aug_acc
divers = torch.norm((pred[val] - aug_pred[val]), p='fro').item()
print('label', aug_acc, divers)


# NASA
aug_g, aug_feat = augmentor.generate('replace', ratio=0.4)
aug_g = aug_g.to(device)
aug_feat = aug_feat.to(device)
aug_h = conv(aug_g, aug_feat)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for _ in range(epoch):
    mlp.train()
    aug_pred = mlp(aug_h)

    with g.local_scope():
        aug_g.ndata['pred'] = aug_pred

        aug_g.update_all(fn.copy_u('pred', '_'), fn.mean('_', 'pred'))
        aug_pred = aug_g.ndata.pop('pred')
        loss = mce_loss(aug_pred.log()[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mlp.eval()
aug_pred = mlp(h)
aug_acc  = accuracy(aug_pred[val], labels[val])

consis = aug_acc
divers = torch.norm((pred[val] - aug_pred[val]), p='fro').item()
print('NASA', aug_acc, divers)


# augmentation
for strategy in ['dropedge', 'dropnode', 'dropout']:
    for i in range(11):
        # generate augmentation
        aug_g, aug_feat = augmentor.generate(strategy, ratio=0.1*i)
        aug_g = aug_g.to(device)
        aug_feat = aug_feat.to(device)
        aug_h = conv(aug_g, aug_feat)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mlp = MLP(feats.size()[1], hidden, nclass, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for _ in range(epoch):

            mlp.train()
            aug_pred = mlp(aug_h)
            loss = mce_loss(aug_pred[train].log(), labels[train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp.eval()
        aug_pred = mlp(h)
        aug_acc  = accuracy(aug_pred[val], labels[val])

        # accuracy evaluation
        consis = aug_acc

        # diversity evaluation
        divers = torch.norm((pred[val] - aug_pred[val]), p='fro').item()

        print(strategy, i/10, consis, divers)

