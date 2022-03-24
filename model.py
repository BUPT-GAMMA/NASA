import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.conv import GraphConv, APPNPConv, GATConv
from dgl.utils import expand_as_pair


class GConv(nn.Module):
    def __init__(self, k=2):
        super(GConv, self).__init__()
        self._k = k

    def forward(self, g, h):
        with g.local_scope():
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for _ in range(self._k):
                h = h * norm
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm

            return h


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)

        return F.softmax(h, 1)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer1 = GATConv(in_dim, hidden_dim, 4, dropout, dropout)
        self.layer2 = GATConv(4*hidden_dim, out_dim, 1, dropout, dropout)

    def forward(self, g, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(g, h).flatten(1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h).mean(1)

        return F.softmax(h, 1)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.layer1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.layer2 = GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True)

    def forward(self, g, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(g, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h)

        return F.softmax(h, 1), h


class GCN_LPA(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout):
        super(GCN_LPA, self).__init__()
        self.dropout = dropout

        self.layer1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.layer2 = GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True)
        self.edge_weight = nn.Parameter(torch.Tensor(g.num_edges(), 1))
        torch.nn.init.uniform_(self.edge_weight)

    def forward(self, g, h, labels, train):
        nclass = labels.size(1)
        emb = torch.zeros((g.num_nodes(), nclass))
        emb = emb.to(labels.device)
        emb[train, :] = labels[train, :]

        for _ in range(20):
            with g.local_scope():
                g.ndata['emb'] = emb
                g.edata['e'] = self.edge_weight
                g.update_all(fn.u_mul_e('emb', 'e', 'm'), fn.mean('m', 'h'))
                emb = g.ndata['h']
            emb[train, :] = labels[train, :]

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(g, h, edge_weight=self.edge_weight.detach())
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h, edge_weight=self.edge_weight.detach())

        return F.log_softmax(h, 1), F.log_softmax(emb, 1)


class APPNP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, dropedge):
        super(APPNP, self).__init__()
        self.dropout = dropout
        self.conv = APPNPConv(k=4, alpha=0.1)
        
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.w1.reset_parameters()
        self.w2.reset_parameters()
    
        self.conv_src = nn.Linear(out_dim, 1, bias=False)
        self.conv_dst = nn.Linear(out_dim, 1, bias=False)
        self.conv_src.reset_parameters()
        self.conv_dst.reset_parameters()


    def forward(self, g, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.w1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.w2(h)
        h = self.conv(g, h)

        with g.local_scope():
            g.srcdata.update({'e_src': self.conv_src(h)})
            g.dstdata.update({'e_dst': self.conv_dst(h)})
            g.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
            e = torch.tanh(g.edata.pop('e'))
            e = edge_softmax(g, e)

        return F.log_softmax(h, 1), e

