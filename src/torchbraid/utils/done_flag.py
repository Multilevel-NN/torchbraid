import torch

__all__ = ['DoneFlag','DoneFlagMixin']

_REGISTER_DONE_FLAG_ATTR = 'register_done'

class DoneFlag:
  @staticmethod
  def allocate():
    """
    Allocate a done flag tensor.
    """
    host_device = torch.device("cpu")
    done_flag = torch.tensor(True,device=host_device)
    return done_flag

  @staticmethod
  def supports(ob):
    """
    Determine if a particle object instance object supports the done flag
    """
    return hasattr(obj,_REGISTER_DONE_FLAG_ATTR)

  @staticmethod
  def obj_register(obj,done_flag):
    """
    Modifies the done flag for this object shares it.
    """
    if hasattr(obj,_REGISTER_DONE_FLAG_ATTR):
      obj.done_flag = done_flag  

  @staticmethod
  def module_register(layer,done_flag):
    """
    Modifies the done flag for all sub modules so the tensor is shared between them.

    After this set has been completed all modifying the persistent tensor "done_flag" 
    will distribute the change to all other modules.
    """
    for l in DoneFlag.modules(layer):
      l.done_flag = done_flag  

  @staticmethod
  def update(done_flag,state):
    """
    Convenience function to change the state
    of a done flag

    done_flag: A 0d tensor with one boolean item, allocateed by alloc_done_flag
    state: True or False, depending  on the desired state of the flag
    """
    done_flag.copy_(state)

  @staticmethod
  def modules(layer):
    """
    Assuming a "torch.nn.Module" build a generator for the for constructing the list
    of those with valid done flag registration.
    """
    return (l for l in layer.modules() if hasattr(l,_REGISTER_DONE_FLAG_ATTR))

class DoneFlagMixin:
  def register_done(self,done_flag):
    """
    Register a done flag to change the behavior.
    """
    assert hasattr(super,_REGISTER_DONE_FLAG_ATTR),\
           '"register_done" was called on an object (inheriting from "DoneFlagMixin")'\
           ' where no "done_flag" attribute was created'
    self.done_flag = done_flag
