import torch
import torch.nn as nn
import torch.nn.functional as F

class LPBatchNorm2d(nn.Module):

  @staticmethod
  def instance_from_module(module):    
    """
    Build a generator finding all sub modules of type LPBatchNorm2d
  
    This method is fully recursive.
    """
    return (l for l in module.modules() if isinstance(l,LPBatchNorm2d))
 
  @staticmethod
  def print_done_flag(module):
    """
    Print the done flag for all sub modules with type LPBatchNorm2d

    This method is fully recursive.
    """
    print([l.done_flag for l in LPBatchNorm2d.instance_from_module(module)])

  @staticmethod
  def set_done_flag(module,done_flag):
    """
    Modifies the done flag for all sub modules so the tensor is shared between them.

    After this set has been completed, all the modules can be updated. 
    """
    for l in LPBatchNorm2d.instance_from_module(module):
      l.done_flag = done_flag  

  @staticmethod
  def alloc_done_flag():
    """
    Convenience function to allocate a done  flag.
    """
    host_device = torch.device("cpu")
    done_flag = torch.tensor(True,device=host_device)
    return done_flag

  @staticmethod
  def update_done_flag(done_flag,state):
    """
    Convenience function to change the state
    of a done flag

    done_flag: A 0d tensor with one boolean item, allocateed by alloc_done_flag
    state: True or False, depending  on the desired state of the flag
    """
    done_flag.copy_(state)
    
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
    done_flag = LPBatchNorm2d.alloc_done_flag()
    self.register_buffer("mean",mean)
    self.register_buffer("var",var)
    self.register_buffer("done_flag",done_flag)
    
  def forward(self,x):
    if self.done_flag and self.training:
        bn_train_mode = True
        mean = self.mean
        var = self.var
    elif not self.done_flag and self.training:
        bn_train_mode = True
        mean = self.mean.clone()
        var = self.var.clone()
    else:
        bn_train_mode = False
        mean = self.mean
        var = self.var
 
    return F.batch_norm(x,mean,var,None,None,bn_train_mode,self.momentum,self.eps)
