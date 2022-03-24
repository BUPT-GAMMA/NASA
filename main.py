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
from model import GCN, APPNP, GAT, GConv, MLP

'''
seed = 9
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

# GCN
# cora 0.01 1e-3 500 32 0.7 1.0 0.5
# cite 0.01 1e-3 500 32 0.1 1.0 0.5
# pubm 0.01 1e-3 500 32 0.5 0.5 0.2
# comp 0.01 1e-3 500 32 0.3 0.7 0.5
# phot 0.01 1e-3 500 32 0.5 0.5 0.5

# less label
# GCN
# cora 0.01 1e-3 500 32 0.8 1.0 0.7 
# cite 0.01 1e-3 500 32 0.8 1.0 1.1
# pubm 0.01 1e-3 500 32 0.5 0.5 0.5


device = 'cuda:1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=1.0, help='loss hyperparameter')
parser.add_argument('--temp', type=float, default=0.5, help='sharpen temperature')
parser.add_argument('--prob', type=float, default=0.5, help='loss hyperparameter')
parser.add_argument('--seed', type=int, default=0, help='Patience')
args = parser.parse_args()

g, nclass = load_data(args.dataset, split='fix')
augmentor = augmentation(g, g.ndata['feat'])
augmentor.init()

g = g.to(device)
feats  = g.ndata['feat']
labels = g.ndata['label']
train  = g.ndata['train_mask']
val    = g.ndata['val_mask']
test   = g.ndata['test_mask']

net = GCN(feats.size()[1], args.hidden, nclass, args.dropout).to(device)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# main loop
dur = []
log = []
counter = 0

bce_loss = nn.BCELoss(reduction='mean')
mce_loss = nn.NLLLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')
kld_loss = nn.KLDivLoss(reduction='batchmean')


def edge_kld_loss(edges):
    src = edges.src['pred'] + 1e-10
    dst = (edges.dst['tar'] + 1e-10).detach()
    loss = (dst*dst.log() - dst*src.log()).sum(1, keepdim=True)
    return {'loss': loss}


def label_prop_loss(edges):
    src = edges.src['pred'] + 1e-10
    dst = (edges.dst['pred'] + 1e-10).detach()
    loss = (dst*dst.log() - dst*src.log()).sum(1, keepdim=True)
    return {'loss': loss}


for epoch in range(args.epochs):
    t0 = time.time()

    aug_g, aug_feat = augmentor.generate(strategy='replace', ratio=args.prob)
    aug_g = aug_g.to(device)
    aug_feat = aug_feat.to(device)

    net.train()
    aug_pred, _ = net(aug_g, aug_feat)
    loss1 = mce_loss((aug_pred + 1e-10).log()[train], labels[train])

    aug_g.ndata['pred'] = aug_pred
    aug_g.update_all(fn.copy_u('pred', '_'), fn.mean('_', 'avg_pred'))
    avg_pred = aug_g.ndata.pop('avg_pred')

    sharp = (torch.pow(avg_pred, 1./args.temp) / torch.sum(torch.pow(avg_pred, 1./args.temp), dim=1, keepdim=True))
    aug_g.ndata['tar'] = sharp

    aug_g.update_all(edge_kld_loss, fn.mean('loss', 'loss'))
    loss2 = aug_g.ndata.pop('loss').mean()

    loss = loss1 + args.alpha * loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    with torch.no_grad():
        pred, _ = net(g, feats)

        train_loss = mce_loss(pred[train].log(), labels[train])
        val_loss = mce_loss(pred[val].log(), labels[val])
        train_acc = accuracy(pred[train], labels[train])
        val_acc = accuracy(pred[val], labels[val])
        test_acc = accuracy(pred[test], labels[test])
        log.append([epoch, val_loss.item(), val_acc, test_acc])

    dur.append(time.time() - t0)

    print("Epoch {:05d} | Train {:.4f} | Valid {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(epoch, train_loss.item(), val_loss.item(), train_acc, val_acc, test_acc, np.mean(dur)))

log.sort(key=lambda x: -x[2])
acc = log[0][-1]
epoch = log[0][0]
print(acc)

