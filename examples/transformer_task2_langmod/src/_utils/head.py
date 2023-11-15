import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
  def __init__(self, head_size, model_dimension, context_window, dropout):
    super().__init__()
    self.key   = nn.Linear(model_dimension, head_size, bias=False)
    self.query = nn.Linear(model_dimension, head_size, bias=False)
    self.value = nn.Linear(model_dimension, head_size, bias=False)
    self.register_buffer(
        'tril', torch.tril(torch.ones(context_window, context_window))
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B,T,C = x.shape
    k = self.key(x)   # (B,T,hs)
    q = self.query(x) # (B,T,hs)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,hs)
    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out
