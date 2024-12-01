from __future__ import print_function

import numpy as np

import sys
import statistics as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI
import math

__all__ = [ 'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

# Define BERT Layers
# Inspired by https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

# We're slowly just porting in the Huggingface stuff

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=64):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(seq_len, embed_size)
        self.norm = nn.LayerNorm(embed_size)

        # We know this isn't dynamic
        seq_len = seq_len
        pos = torch.arange(seq_len, dtype=torch.long)
        # pos = pos.unsqueeze(0).expand_as(sequence)  # (seq_len,) -> (batch_size, seq_len)
        self.register_buffer("pos", pos)

    def forward(self, sequence, segment_label):
        # x = self.token(sequence) + self.position(pos) + self.segment(segment_label)
        x = self.token(sequence) + self.position(self.pos.unsqueeze(0).expand_as(sequence)) + self.segment(segment_label)
        return self.norm(x)

### attention layers
class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, middle_dim=2048, dropout=0.0):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

####################################################################################
####################################################################################
# Network architecture is Open + BERT Encoders + multiple closing layers

class OpenLayer(nn.Module):
    """
    Simply embeds and adds positional encodings
    """
    def __init__(self, vocab_size, d_model=768, seq_len=64):
        super(OpenLayer, self).__init__()
        self.d_model = d_model
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len)

    def forward(self, x, segment_info):
        # attention masking for padded token

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        return x

class CloseLayerNSP(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """
    def __init__(self, hidden):
        super(CloseLayerNSP, self).__init__()
    
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class CloseLayerMLM(nn.Module):
    """
        predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        super(CloseLayerMLM, self).__init__()

        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class StepLayer(nn.Module):
    """
    Just the encoder layer 
    """
    def __init__(self, model_dimension, num_heads):
        super(StepLayer, self).__init__()

        self.d_model = model_dimension
        self.num_heads = num_heads

        self.feed_forward = FeedForward(self.d_model, middle_dim=self.d_model * 4)
        self.mha = MultiHeadedAttention(self.num_heads, self.d_model, dropout=0.0)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        # super(EncoderLayer, self).__init__()
        # self.layernorm = torch.nn.LayerNorm(d_model)
        # self.self_multihead = MultiHeadedAttention(heads, d_model)
        # self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        # self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, dt: float=1.0):    # Default must be there for shape
        # Need to use global mask; passing in stuff might be hard
        global mask

        # Apply multi-step version
        x1 = self.ln1(x)
        x1 = x + dt * self.mha(
            x1, x1, x1, mask
        )

        x2 = self.ln2(x1)
        x2 = x1 + dt * self.feed_forward(x2)
        
        return x2

####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
    def __init__(self, vocab_size, model_dimension=768, num_heads=12, seq_len=64,
        local_steps=8, Tf=1.0,max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
        super(ParallelNet, self).__init__()

        step_layer = lambda: StepLayer(model_dimension, num_heads)
        self.comm_lp = comm_lp
        numprocs = self.comm_lp.Get_size()

        # Seperate max_levels to forward and backawrd
        if not isinstance(max_levels, int):
            max_fwd_levels = max_levels[0]
            max_bwd_levels = max_levels[1]
        else:
            max_fwd_levels = max_levels
            max_bwd_levels = max_levels


        self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, local_steps*numprocs, Tf,
                                                    max_fwd_levels=max_fwd_levels, max_bwd_levels=max_bwd_levels,
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

        # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels)
        # on processors not equal to 0, these will be None (there are no parameters to train there)
        self.open_nn = compose(OpenLayer, vocab_size, model_dimension, seq_len)
        self.close_nn_nsp = compose(CloseLayerNSP, model_dimension)
        self.close_nn_mlm = compose(CloseLayerMLM, model_dimension, vocab_size)

    def saveSerialNet(self, name):
        # Model can be reloaded in serial format with: model = torch.load(filename)
        serial_nn = self.parallel_nn.buildSequentialOnRoot()
        if self.comm_lp.Get_rank() == 0:
            s_net = SerialNet(-1, -1, -1, 
                              serial_nn=serial_nn, open_nn=self.open_nn, close_nn_nsp=self.close_nn_nsp, close_nn_mlm=self.close_nn_mlm)
            s_net.eval()
            torch.save(s_net, name)

    def forward(self, x, segment_info):
        # We need the mask to be passed through, so using global allows us 
        # to pass through without interfering with the parallel_nn
        
        # Start with (bs, seq_len) -> unsqueeze -> (bs, 1, seq_len) -> repeat -> (bs, seq_len, seq_len) -> unsqueeze -> (bs, 1, seq_len, seq_len)
        global mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
        # this makes sure this is run on only processor 0
        x = self.compose(self.open_nn, x, segment_info)

        # TODO: output mask to make sure this is fine
        x = self.parallel_nn(x)

        # Go through the classifier layer
        nsp = self.compose(self.close_nn_nsp, x)
        mlm = self.compose(self.close_nn_mlm, x)

        return nsp, mlm

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
  def __init__(self, channels=12, local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn_nsp=None, close_nn_mlm=None):
    super(SerialNet, self).__init__()

    self.open_nn = open_nn

    self.serial_nn = serial_nn

    self.close_nn_nsp = close_nn_nsp
    self.close_nn_mlm = close_nn_mlm

  def forward(self, x, segment_info):
    global mask
    mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    x = self.open_nn(x, segment_info)
    x = self.serial_nn(x)
    nsp = self.close_nn_nsp(x)
    mlm = self.close_nn_mlm(x)
    return nsp, mlm

