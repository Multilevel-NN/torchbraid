import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


def getConnectivity(X, nsparse=16):
    X2 = torch.pow(X, 2).sum(dim=1, keepdim=True)
    D = X2 + X2.transpose(2, 1) - 2 * X.transpose(2, 1) @ X
    D = torch.exp(torch.relu(D))

    vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)
    IJ = torch.stack([I, J], dim=1)

    return IJ


def makeBatch(Ilist, Jlist, nnodesList, Wlist=[1.0]):
    I = torch.tensor(Ilist[0])
    J = torch.tensor(Jlist[0])
    W = torch.tensor(Wlist[0])
    nnodesList = torch.tensor(nnodesList, dtype=torch.long)
    n = nnodesList[0]
    for i in range(1, len(Ilist)):
        Ii = torch.tensor(Ilist[i])
        Ji = torch.tensor(Jlist[i])

        I = torch.cat((I, n + Ii))
        J = torch.cat((J, n + Ji))
        ni = nnodesList[i].long()
        n += ni
        if len(Wlist) > 1:
            Wi = torch.tensor(Wlist[i])
            W = torch.cat((W, Wi))

    return I, J, nnodesList, W


class graph(nn.Module):

    def __init__(self, iInd, jInd, nnodes, W=torch.tensor([1.0]), pos=None, faces=None):
        super(graph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        self.W = W.to(device)
        self.pos = pos
        self.faces = faces

    def nodeGrad(self, x, W=[]):
        if len(W) == 0:
            W = self.W
        if W.shape[0] == x.shape[2]:
            # if its degree matrix
            g = W[self.iInd] * (x[:, :, self.iInd] - x[:, :, self.jInd])
        else:
            g = W * (x[:, :, self.iInd] - x[:, :, self.jInd])

        return g

    def nodeAve(self, x, W=[]):
        if len(W) == 0:
            W = self.W
        if W.shape[0] == x.shape[2]:
            g = W[self.iInd] * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        else:
            g = W * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        return g

    def edgeDiv(self, g, W=[]):
        if len(W) == 0:
            W = self.W
        x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # for i in range(self.iInd.numel()):
        #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # for j in range(self.jInd.numel()):
        #    x[:,:,self.jInd[j]] -= w*g[:,:,j]
        if W.shape[0] != g.shape[2]:
            x.index_add_(2, self.iInd, W[self.iInd] * g)
            x.index_add_(2, self.iInd, W[self.jInd] * g)
        else:
            x.index_add_(2, self.iInd, W * g)
            x.index_add_(2, self.jInd, -W * g)

        return x

    def edgeAve(self, g, method='max', W=[]):
        if len(W) == 0:
            W = self.W
        x1 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        if W.shape[0] != g.shape[2]:
            x1.index_add_(2, self.iInd, W[self.iInd] * g)
            x2.index_add_(2, self.jInd, W[self.jInd] * g)
        else:
            x1.index_add_(2, self.iInd, W * g)
            x2.index_add_(2, self.jInd, W * g)
        if method == 'max':
            x = torch.max(x1, x2)
        elif method == 'ave':
            x = (x1 + x2) / 2
        return x

    def nodeLap(self, x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        return L


class dense_graph(nn.Module):

    def __init__(self, nnodes, W=torch.ones(1)):
        super(dense_graph, self).__init__()
        self.nnodes = nnodes
        self.W = W

    def nodeGrad(self, x, W=[]):
        if len(W) == 0:
            W = self.W
        x = x.squeeze(0).unsqueeze(1)
        g = W * (x - x.transpose(1, 2))
        g = g.unsqueeze(0)
        return g

    def nodeAve(self, x, W=[]):
        if len(W) == 0:
            W = self.W
        x = x.squeeze(0).unsqueeze(1)
        g = W * (x + x.transpose(1, 2)) / 2.0
        g = g.unsqueeze(0)
        return g

    def edgeDiv(self, g, W=[]):
        if len(W) == 0:
            W = self.W
        g = W * g
        x1 = g.sum(dim=2, keepdim=True).squeeze(0)
        x2 = g.sum(dim=3, keepdim=True).squeeze(0)
        x = x1 - x2.transpose(2, 1)
        x = x.squeeze(1).unsqueeze(0)
        return x

    def edgeAve(self, g, method='max', W=[]):
        if len(W) == 0:
            W = self.W
        g = W * g
        x1 = g.mean(dim=2, keepdim=True).squeeze(0)
        x2 = g.mean(dim=3, keepdim=True).squeeze(0)
        x2 = x2.transpose(1, 2)
        if method == 'max':
            x = torch.max(x1, x2)
        else:
            x = (x1 + x2) / 2
        x = x.squeeze(1).unsqueeze(0)
        return x

    def nodeLap(self, x, W=[]):
        if len(W) == 0:
            W = self.W
        g = self.nodeGrad(x, W)
        d = self.edgeDiv(g, W)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = torch.pow(g, 2).sum(dim=1)
        return L


### Try to work in parallel
test = False
if test:
    G = []
    x = []
    for i in range(1000):
        n = torch.randint(100, 400, (1,))
        Gi = dense_graph(n)
        xi = torch.rand(1, 10, n)
        G.append(Gi)
        x.append(xi)

    t1 = time.time()
    tt = list(map(dense_graph.nodeGrad, G, x))
    t2 = time.time()
    print(t2 - t1)

    tt = []
    t1 = time.time()
    for i in range(len(G)):
        ti = G[i].nodeGrad(x[i])
        tt.append(ti)
    t2 = time.time()
    print(t2 - t1)

###### Testing stuff
# tests = 1
# if tests:
#    nnodes = 512
#     II = torch.torch.zeros(nnodes*(nnodes-1)//2)
#     JJ = torch.torch.zeros(nnodes*(nnodes-1)//2)
#
#     k = 0
#     for i in range(nnodes):
#         for j in range(i+1,nnodes):
#             II[k] = i
#             JJ[k] = j
#             k+=1
#
#     G = graph(II,JJ,nnodes)
#     x  = torch.randn(1,128,nnodes)
#
#     test_adjoint = 0
#     if test_adjoint:
#         # Adjoint test
#         w = torch.rand(G.iInd.shape[0])
#         y = G.nodeGrad(x,w)
#         ne = G.iInd.numel()
#         z = torch.randn(1,128,ne)
#         a1 = torch.sum(z*y)
#         v = G.edgeDiv(z,w)
#         a2 = torch.sum(v*x)
#         print(a1,a2)
#
#
#
#     nhid = 8
#     L = graphDiffusionLayer(G,x.shape[1],nhid)
#
#     y = L(x)
