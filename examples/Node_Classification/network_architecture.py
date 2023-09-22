from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI

import graphOps as GO
from graphOps import getConnectivity
from mpl_toolkits.mplot3d import Axes3D
from utils import saveMesh
from inits import glorot, identityInit

__all__ = [ 'OpenFlatLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

####################################################################################
####################################################################################
# Network architecture is Open + ResNet + Close
# The StepLayer defines the ResNet (ODENet)

class OpenFlatLayer(nn.Module):
  ''' 
  Opening layer has no parameters, replicates image number of channels times
  '''
  def __init__(self, nopen,nNin,dropout=False,realvarlet=False,faust = False,PPI=False):
    super(OpenFlatLayer, self).__init__()
    self.dropout = dropout
    self.realVarlet = realvarlet
    self.faust = faust
    self.PPI = PPI
    stdv = 1e-2
    if self.faust or self.PPI:
        stdv = 1e-1

    self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
    self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

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
            xe = conv1(xe, K)
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
  
  def forward(self, xn,**extra_kwargs):
    # Opening layer
    # xn = [B, C, N]
    # xe = [B, C, N, N] or [B, C, E]
    # Opening layer
    
    [Graph, edge_index] = self.updateGraph(extra_kwargs["Graph"])
    xhist = []

    # dropout layer (n x cin -> n x cin)
    if self.dropout:
      xn = F.dropout(xn, p=self.dropout, training=self.training)

    # 1 x 1 convolution (n x cin -> n x c)
    # and ReLU layer
    xn = self.singleLayer(xn, self.K1Nopen, relu=True, openclose=True, norm=False)
    return xn , Graph

class CloseLayer(nn.Module):
  '''c
  Dense closing classification layer
  '''
  def __init__(self,nopen,num_output,dropOut=False,modelnet=False,faust=False,PPI=False):
    super(CloseLayer, self).__init__()
    self.modelnet = modelnet
    self.faust = faust
    self.PPI = PPI
    if dropOut > 0.0:
        self.dropout = dropOut
    else:
        self.dropout = False
    stdv = 1e-2
    if self.faust or self.PPI:
        stdv = 1e-1
        
    self.KNclose = nn.Parameter(torch.randn(nopen, nopen) * stdv)
    if self.modelnet:
        self.KNclose = nn.Parameter(torch.randn(1024, num_output) * stdv)  # num_output on left size
    elif not self.faust:
        self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  # num_output on left size

  def forward(self, xn):
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
      return xn

    ## Otherwise its citation graph node classification:
    return F.log_softmax(xn, dim=1)

# conv2: performs a 2D convolution operation using the F.conv2d
# X: input graph
# Kernel: convolution kernel or filter. 
# It is a tensor that defines the weights of the convolution operation.
def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
# padding: ensures that the output has the same spatial dimensions as the input.


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

# Use device or CPU?
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


####################################################################################
####################################################################################
# Network architecture is Encoder + PDE-GCN Layers + Decoder
# The StepLayer defines the PDE-GCN Layers

class StepLayer(nn.Module):
  '''
  ResNet composed of convolutional layers
  '''
  def __init__(self,nopen, nhid, dropOut,realVarlet,varlet,gated,doubleConv,h):
    super(StepLayer, self).__init__()
    if dropOut > 0.0:
        self.dropout = dropOut
    else:
        self.dropout = False

    stdvp = 1e-2    
    self.h = h
    self.realVarlet = realVarlet
    self.varlet = varlet
    self.gated = gated
    self.doubleConv = doubleConv
    if varlet:
        Nfeatures = 1 * nopen
    else:
        Nfeatures = 1 * nopen

    self.KN1 = nn.Parameter(torch.rand(1, Nfeatures, nhid) * stdvp)
    rrnd = torch.rand( Nfeatures, nhid) * (1e-3)

    self.KN1 = nn.Parameter(identityInit(self.KN1) + rrnd)

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
            xe = conv1(xe, K)
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
      
  def forward(self, xn,*extra_args,**extra_kwargs): # Forward for step layer
    gradX = extra_kwargs["Graph"].nodeGrad(xn)

    if self.dropout:
        gradX = F.dropout(gradX, p=self.dropout, training=self.training)

    # 1 x 1 convolution
    dxn = (self.singleLayer(gradX, self.KN1[0], norm=False, relu=True, groups=1))  # KN2
    dxn = extra_kwargs["Graph"].edgeDiv(dxn)
    return -self.h*dxn


####################################################################################
####################################################################################

# Parallel Graphn Network class
# local_steps: number of PDE_GCN layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelGraphNet(nn.Module):
  def __init__(self,nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
              diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
              graphUpdate=None, PPI=False, gated=False, realVarlet=False, mixDyamics=False, doubleConv=False,
              tripleConv=False,channels=12, Tf=1.0, max_levels=1, bwd_max_iters=1,
              fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
              fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
              user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
    super(ParallelGraphNet, self).__init__()
    # PDE-GCN Parameters
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

    self.PPI = PPI
    # Parallel Net Parameters
    step_layer = lambda: StepLayer(nopen, nhid, dropOut,realVarlet,varlet,gated,doubleConv,h)
    self.comm_lp = comm_lp
    numprocs = self.comm_lp.Get_size()
    local_steps = self.nlayers/numprocs
    self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, self.nlayers, Tf,
                                                max_fwd_levels=max_levels, max_bwd_levels=max_levels,
                                                max_iters=2, user_mpi_buf=user_mpi_buf)
    self.parallel_nn.setBwdMaxIters(bwd_max_iters)
    self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level, True)
    self.parallel_nn.setPrintLevel(braid_print_level, False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(relax_only_cg)
    if fmg:
      self.parallel_nn.setFMG()

    self.parallel_nn.setNumRelax(1)  # FCF relaxation default on coarse levels
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0, level=0)  # Set F-Relaxation only on the fine grid
    else:
      self.parallel_nn.setNumRelax(1, level=0)  # Set FCF-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()

    # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels)
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenFlatLayer,nopen,nNin,dropOut,realVarlet)
    self.close_nn = compose(CloseLayer, nopen,num_output,dropOut,modelnet,faust,PPI)

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


  def forward(self, xn,*extra_args,**extra_kwargs):
    # Opening layer
    # xn = [B, C, N]
    # xe = [B, C, N, N] or [B, C, E]
    # Opening layer
    try:
      xn,Graph = self.compose(self.open_nn,xn,Graph=extra_kwargs["Graph"])
    except:
      [Graph, edge_index] = self.updateGraph(extra_kwargs["Graph"])
    xn = self.parallel_nn(xn,[],Graph=Graph)
    xn = self.compose(self.close_nn,xn)
    return xn # For backpropagation xn_0 must be set on in ranks not equat to 0

    """# dropout and 1 x 1 convolution layer
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
      return xn"""




# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialGraphNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None):
    super(SerialGraphNet, self).__init__()

    if open_nn is None:
      self.open_nn = EncoderLayer(channels)
    else:
      self.open_nn = open_nn

    if serial_nn is None:
      step_layer = lambda: StepLayer(channels)
      numprocs = 1
      parallel_nn = torchbraid.LayerParallel(MPI.COMM_SELF, step_layer, numprocs * local_steps, Tf,
                                             max_fwd_levels=1, max_bwd_levels=1, max_iters=1)
      parallel_nn.setPrintLevel(0, True)
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if close_nn is None:
      self.close_nn = DecoderLayer(channels)
    else:
      self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x


####################################################################################
####################################################################################

# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='Node Classification example argument parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--dataset', default='CORA',
                      help='dataset can be either CORA, Pubmed and Citeseer')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the resnet layer (default: 32)')
  parser.add_argument('--channels', type=int, default=3, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for ResNet layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)') #TODO Change
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel max number of levels (default: 3)')
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel max backward iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel max forward iterations (default: 2)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-fine-fcf',action='store_true', default=False,
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--warm-up', action='store_true', default=False,
                      help='Warm up for GPU timings (default: False)')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')

  # data parallelism
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size != 0:
    root_print(rank, 1, 1, 'Data parallel size must be an even multiple of the number of processors: %d %d'
               % (procs, args.dp_size) )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  ##
  # Compute number of parallel-in-time multigrid levels 
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(args.steps, min_coarse_size, args.lp_cfactor)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################
