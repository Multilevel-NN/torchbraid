from __future__ import print_function

import numpy as np

import sys
import statistics as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid

from mpi4py import MPI
import math
from dataclasses import dataclass
from torchbraid.utils import LPDropout as Dropout

__all__ = [ 'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

# Ported miniGPT layers
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



####################################################################################
####################################################################################
# Network architecture is Open + GPT2 Dencoders + Classifier layers with layer norm


class OpenLayer(nn.Module):
    """
    Just the embedding
    """
    def __init__(self, config):
        super(OpenLayer, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = Dropout(config.dropout)
        self.config = config

        if config.buffer_layers:
            self.layer1 = StepLayer(config)
            self.layer2 = StepLayer(config)
          
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def forward(self, idx):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        if self.config.buffer_layers: 
            x = self.layer1(x, dt=1)
            x = self.layer2(x, dt=1)

        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CloseLayer(nn.Module):
    """
    Close; final layer norm and classifier
    """
    def __init__(self, config):
        super(CloseLayer, self).__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # self.ln_f = nn.Identity()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        if config.buffer_layers:
            self.layer1 = StepLayer(config)
            self.layer2 = StepLayer(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, decoder_output):
        if self.config.buffer_layers: 
            decoder_output = self.layer1(decoder_output, dt=1)
            decoder_output = self.layer2(decoder_output, dt=1)

        sequence_output = self.ln_f(decoder_output)
        logits = self.lm_head(sequence_output)
        # print(logits.shape) # torch.Size([64, 512, 50304])
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class StepLayer(nn.Module):
    """
    Just the decoder layer
    """
    def __init__(self, config):
        super(StepLayer, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        if config.periln:
            self.ln_11 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_22 = LayerNorm(config.n_embd, bias=config.bias)
        self.config = config
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def forward(self, x, dt: float=1.0):
        if self.config.periln:
            x = x + dt/2 * self.ln_11(self.attn(self.ln_1(x)))
            x = x + dt/2 * self.ln_22(self.mlp(self.ln_2(x)))
        else:
            x = x + dt/2 * self.attn(self.ln_1(x))
            x = x + dt/2 * self.mlp(self.ln_2(x))

        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

####################################################################################
####################################################################################

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    buffer_layers: bool = False
    periln: bool = False


# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
    def __init__(self, config = None,
        local_steps=8, Tf=1.0,max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
        super(ParallelNet, self).__init__()

        step_layer = lambda: StepLayer(config)
        self.comm_lp = comm_lp
        numprocs = self.comm_lp.Get_size()

        # Use the same config (except for hidden layers which is contorlled elsewhere)
        # as bert-large-uncased
        if config is None: 
            config = GPTConfig(
                n_layer=local_steps * numprocs, # Total number of layers 
                bias=False                      # SUpposedly easier to train
            )

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
        self.parallel_nn.setBwdResidualCompute(True)
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
        self.open_nn = compose(OpenLayer, config)
        self.close_nn = compose(CloseLayer, config)

    def saveSerialNet(self, name):
        # Model can be reloaded in serial format with: model = torch.load(filename)
        serial_nn = self.parallel_nn.buildSequentialOnRoot()
        if self.comm_lp.Get_rank() == 0:
            s_net = SerialNet(serial_nn=serial_nn, open_nn=self.open_nn,
                              close_nn=self.close_nn)
            s_net.eval()
            torch.save(s_net, name)

    def forward(self, x):
        x = self.compose(self.open_nn, x)
        x = self.parallel_nn(x)
        out = self.compose(self.close_nn, x)

        return out
    
    # Let's not do dropout for this yet
    # 
    def new_mask(self, x=None): 
      """
      Generates a new mask given a sample input
      """
      raise Exception
      for layer in self.parallel_nn.local_layers.modules():
        # print(layer)
        if isinstance(layer, Dropout):
          # print('Generating')
          layer.generate_new_mask(x)

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
    def __init__(self, serial_nn=None, open_nn=None, close_nn=None):
        super(SerialNet, self).__init__()

        self.open_nn = open_nn
        self.serial_nn = serial_nn
        self.close_nn = close_nn

    def forward(self, x):
        x = self.open_nn(x)
        x = self.serial_nn(x)
        out = self.close_nn(x)

        return out
