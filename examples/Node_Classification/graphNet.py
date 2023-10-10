import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
from torch_geometric.nn import global_max_pool

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin

try:
    from src import graphOps as GO
    from src.graphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from src.utils import saveMesh, h_swish
    from src.inits import glorot, identityInit

except:
    import graphOps as GO
    from graphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from utils import saveMesh
    from inits import glorot, identityInit

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCN2Conv
from torch_scatter import scatter_add

# conv2: performs a 2D convolution operation using the F.conv2d
# X: input graph
# Kernel: convolution kernel or filter. 
#         It is a tensor that defines the weights of the convolution operation.
def compose(f,a1,a2):
    return f(a1,a2)

def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
# padding: ensures that the output has the same spatial dimensions as the input.


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#  total variation (TV) normalization of a given input tensor X
# X: input
# eps: a small constant (default value: 1e-3) added to the denominator 
# to prevent division by zero.
def tv_norm(X, eps=1e-3):
    # centers the data by subtracting the mean of each feature or channel.
    X = X - torch.mean(X, dim=1, keepdim=True)
    # normalizes the data by dividing each feature or channel by its standard deviation.
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X

# difference along the second dimension of the input tensor X
# used to find gradient of the input tensor along the specified dimension.
def diffX(X):
    X = X.squeeze()
    return X[:, 1:] - X[:, :-1]


def diffXT(X):
    X = X.squeeze()
    D = X[:, :-1] - X[:, 1:]
    # subtracts the elements at index i+1 from the elements at index i 
    # for each row in X, resulting in a tensor D with one fewer column.

    # Handling Boundary Elements:
    # Extracts the first column of X and negates it
    d0 = -X[:, 0].unsqueeze(1)
    # Extracts the last column of X and keeps it as it is. 
    d1 = X[:, -1].unsqueeze(1)
    # Concatenates d0, D, and d1 along the second dimension
    D = torch.cat([d0, D, d1], dim=1)
    return D

# x: input
# K1,K2: first and second convolution kernel
def doubleLayer(x, K1, K2):
    # 1D convolution operation using the kernel K1 on the input tensor x. 
    # The unsqueeze(-1) function adds a singleton dimension at the end of
    # K1 to match the required shape for a 1D convolution.
    x = F.conv1d(x, K1.unsqueeze(-1))
    # Layer normalization normalizes the values across the second dimension 
    # (channels) of x based on its shape.
    x = F.layer_norm(x, x.shape)
    # Applies the ReLU activation function element-wise to the tensor x, 
    # which sets all negative values to zero.
    x = torch.relu(x)
    x = F.conv1d(x, K2.unsqueeze(-1))
    return x


###################################################################################pdegcn

# The function constructs an MLP by iterating over the channels 
# list and creating a sequence of fully connected layers,
# channels: list of integers representing the number of output channels in each layer
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class graphNetwork_nodesOnly(nn.Module):

    def __init__(self, nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
                 graphUpdate=None, PPI=False, gated=False, realVarlet=False, mixDyamics=False, doubleConv=False,
                 tripleConv=False):
        super(graphNetwork_nodesOnly, self).__init__()
        self.wave = wave
        self.realVarlet = realVarlet
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.mixDynamics = mixDyamics
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_output = num_output
        self.graphUpdate = graphUpdate
        self.doubleConv = doubleConv
        self.tripleConv = tripleConv
        self.gated = gated
        self.faust = faust
        self.PPI = PPI
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        self.nlayers = nlayer
        stdv = 1e-2
        stdvp = 1e-2
        if self.faust or self.PPI:
            stdv = 1e-1
            stdvp = 1e-1
            stdv = 1e-2
            stdvp = 1e-2
        # nopen: output channel
        # nNin: input channel
        if varlet:
            Nfeatures = 1 * nopen
        else:
            Nfeatures = 1 * nopen
            
        self.KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)
        rrnd = torch.rand(nlayer, Nfeatures, nhid) * (1e-3)

        self.KN1 = nn.Parameter(identityInit(self.KN1))
        """if self.realVarlet:
            self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)

        if self.mixDynamics:
            self.alpha = nn.Parameter(-0 * torch.ones(1, 1))

        if self.faust:
            self.lin1 = torch.nn.Linear(nopen, nopen)
            self.lin2 = torch.nn.Linear(nopen, num_output)

        self.modelnet = modelnet

        self.PPI = PPI
        if self.modelnet:
            self.mlp = Seq(
                MLP([64, 128]), Dropout(0.5), MLP([128, 64]), Dropout(0.5),
                Lin(64, 10))"""

        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.convs1x1 = nn.Parameter(torch.randn(nlayer, nopen, nopen) * stdv)
        self.modelnet = modelnet
        self.compose = compose

        if self.modelnet:
            self.KNclose = nn.Parameter(torch.randn(1024, num_output) * stdv)  # num_output on left size
        elif not self.faust:
            self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  # num_output on left size
        else:
            self.KNclose = nn.Parameter(torch.randn(nopen, nopen) * stdv)

    def reset_parameters(self):
        # glorot: calculates the appropriate standard deviation based 
        # on the tensor's shape and then initializes the tensor with 
        # random values from a uniform distribution.
        glorot(self.K1Nopen)
        glorot(self.K2Nopen)
        glorot(self.KNclose)
        if self.realVarlet:
            glorot(self.KE1)
        if self.modelnet:
            glorot(self.mlp)

    def edgeConv(self, xe, K, groups=1):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
            else:
                xe = conv2(xe, K, groups=groups)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            else:
                xe = conv1(xe, K, groups=groups)
        return xe

    def singleLayer(self, x, K, relu=True, norm=False, groups=1, openclose=False):
        if openclose:  # if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if norm:
                x = F.instance_norm(x)
            if relu:
                # relu layer
                x = F.relu(x)
            else:
                x = F.tanh(x)
        if not openclose:  # if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if not relu:
                x = F.tanh(x)
            else:
                x = F.relu(x)
            if norm:
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t(), groups=groups)
        return x

    def finalDoubleLayer(self, x, K1, K2):
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def savePropagationImage(self, xn, Graph, i=0, minv=None, maxv=None):
        plt.figure()
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        if (maxv is not None) and (minv is not None):
            plt.imshow(img, vmax=maxv, vmin=minv)
        else:
            plt.imshow(img)

        plt.colorbar()
        plt.show()
        plt.savefig('plots/layer' + str(i) + '.jpg')

        plt.close()

    def updateGraph(self, Graph, features=None):
        # If features are given - update graph according to feaure space l2 distance
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        # edge_index is in (2, num_edges) now

        if features is not None:
            features = features.squeeze()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)
            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)

        else:
            [edge_index, edge_weights] = gcn_norm(edge_index)  # Pre-process GCN normalization.
            I = edge_index[0, :]
            J = edge_index[1, :]
            # deg = self.getDegreeMat(Graph)
            Graph = GO.graph(I, J, N, W=edge_weights, pos=None, faces=None)

        return Graph, edge_index

    def forward(self, xn, Graph, data=None, xe=None):
        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        print("Opening Input")
        print(xn)
        if not self.faust:
            [Graph, edge_index] = self.updateGraph(Graph)
        if self.faust:
            xn = torch.cat([xn, Graph.edgeDiv(xe)], dim=1)
        xhist = []
        debug = False

        if debug:
            xnnorm = torch.norm(xn, dim=1)
            vmin = xnnorm.min().detach().numpy()
            vmax = xnnorm.max().detach().numpy()
            saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, -1, vmax=vmax, vmin=vmin)

        if self.realVarlet:
            xe = Graph.nodeGrad(xn)
            if self.dropout:
                xe = F.dropout(xe, p=self.dropout, training=self.training)
            xe = self.singleLayer(xe, self.K2Nopen, relu=True)

        # dropout layer (n x cin -> n x cin)
        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)
        # 1 x 1 convolution (n x cin -> n x c)
        # and ReLU layer
        xn = self.singleLayer(xn, self.K1Nopen, relu=True, openclose=True, norm=False)
        
        x0 = xn.clone()
        debug = False

        xn_old = x0
        print("Opening Output")
        print(xn)
        nlayers = self.nlayers
        for i in range(nlayers):
            print("Input ",i)
            print(xn)
            gradX = Graph.nodeGrad(xn) # Apply G to node features

            if self.dropout:
                if self.varlet:
                    gradX = F.dropout(gradX, p=self.dropout, training=self.training) # This updates gradX, not in equation
            if self.varlet and not self.gated:
                if not self.doubleConv:
                    # 1 x 1 convolution
                    dxn = (self.singleLayer(gradX, self.KN1[i], norm=False, relu=True, groups=1))  # Apply 1x1 convolution + nonlinearity + 1x1 convolution (with same kernel K)
                else:
                    dxn = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i]) # Apply 1x1 convolution + nonlinearity + 1x1 convolution
                dxn = Graph.edgeDiv(dxn) # Applies final G^t 
                        # false
                        # if self.tripleConv:
                        #     dxn = self.singleLayer(dxn, self.KN3[i], norm=False, relu=False)
                    # else: # false
                    #     if not self.doubleConv:
                    #         dxe = (self.singleLayer(gradX, self.KN1[i], norm=False, relu=False, groups=1))
                    #     else:
                    #         dxe = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i])
                    #     dxn = Graph.edgeDiv(dxe)
                    #     if self.tripleConv:
                    #         dxn = self.singleLayer(dxn, self.KN3[i], norm=False, relu=False)
                # false
                # elif self.varlet and self.gated:
                #     W = F.tanh(Graph.nodeGrad(self.singleLayer(xn, self.KN2[i], relu=False)))
                #     lapX = Graph.nodeLap(xn)
                #     dxn = F.tanh(lapX + Graph.edgeDiv(W * Graph.nodeGrad(xn)))
                # else:
                #     dxn = (self.singleLayer(lapX, self.KN1[i], relu=False))
                #     dxn = F.tanh(dxn)

                # true
                if self.mixDynamics:
                    tmp_xn = xn.clone()
                    beta = F.sigmoid(self.alpha)
                    alpha = 1 - beta

                    if 1 == 1:
                        alpha = alpha / self.h
                        beta = beta / (self.h ** 2)

                        xn = (2 * beta * xn - beta * xn_old + alpha * xn - dxn) / (beta + alpha)
                    else:
                        alpha = 0.5 * alpha / self.h
                        beta = beta / (self.h ** 2)

                        xn = (2 * beta * xn - beta * xn_old + alpha * xn_old - dxn) / (beta + alpha)
                    xn_old = tmp_xn

                elif self.wave:
                    tmp_xn = xn.clone()
                    xn = 2 * xn - xn_old - (self.h ** 2) * dxn
                    xn_old = tmp_xn
                else:
                    tmp = xn.clone()
                    xn = (xn - self.h * dxn)
                    xn_old = tmp

                if self.modelnet:
                    xhist.append(xn)

            # if debug:
            #     if image:
            #         self.savePropagationImage(xn, Graph, i + 1, minv=minv, maxv=maxv)
            #     else:
            #         saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1, vmax=vmax, vmin=vmin)

        # dropout and 1 x 1 convolution layer
        xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))

        xn = xn.squeeze().t()
        if self.modelnet:
            out = global_max_pool(xn, data.batch)
            out = self.mlp(out)
            return F.log_softmax(out, dim=-1)

        if self.faust:
            x = F.elu(self.lin1(xn))
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1), F.sigmoid(self.alpha)

        if self.PPI:
            return xn, Graph

        ## Otherwise its citation graph node classification:
        return F.log_softmax(xn, dim=1), Graph
