import torch
import torch.nn as nn
import torch.nn.functional as F

from .done_flag import *

__all__ = ['LPBatchNorm2d']

class LPBatchNorm2d(DoneFlagMixin,nn.Module):
    
  def __init__(self,channels,momentum=0.1,eps=1e-5):
    """
    Constructor. This is the same as BatchNorm2d, however it has additional functionality
    that allows the user to set the internal done_flag tensor. This done flag is anticipated
    to be paired with training and using 'done' in the layer-parallel/MGRIT algorithm. 
    The default value of the done flag is True. In that case, the batch norm during evaluation
    and training will behave like torch.nn.BatchNorm2d. However, when the done flag is False,
    training will not result in a permenant change in running mean and variance (eval has the
    same behavior).

    channels: See torch.nn.BatchNorm2d
    momentum: See torch.nn.BatchNorm2d
    eps: See torch.nn.BatchNorm2d
    """
    super().__init__()
    self.channels = channels
    self.eps      = eps
    self.momentum = momentum
    
    mean = torch.zeros(self.channels)
    var = torch.ones(self.channels)
    done_flag = DoneFlag.allocate()
    self.register_buffer("mean",mean)
    self.register_buffer("var",var)
    self.register_buffer("done_flag",done_flag)

  def reset_running_stats(self):
    self.mean.zero_()
    self.var.fill_(1)

  def forward(self,x):
    if self.done_flag and self.training:
      bn_train_mode = True
      mean = self.mean
      var = self.var
      #print('TRAIN-DONE')
    elif not self.done_flag and self.training:
      bn_train_mode = True
      mean = self.mean.clone()
      var = self.var.clone()
      #print('TRAIN-FIX')
    else:
      bn_train_mode = False
      mean = self.mean
      var = self.var
      #print('EVAL')
 
    return F.batch_norm(x,mean,var,None,None,bn_train_mode,self.momentum,self.eps)
