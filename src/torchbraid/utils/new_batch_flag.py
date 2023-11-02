"""
Utility counter to determine when a new batch is used mainly for 
the Dropout layer. We use the shorthand NB for new batch. 

Should really rename to NBCounter ... 
"""

import torch

__all__ = ['NBFlag', 'NBFlagMixin']

_REGISTER_NB_FLAG_ATTR = 'register_nb'

class NBFlag:
  @staticmethod
  def allocate():
    """
    Allocate a new batch (NB) flag tensor on CPU only
    """
    host_device = torch.device('cpu')
    nb_flag = torch.tensor(1, device=host_device)
    return nb_flag

  @staticmethod
  def obj_register(obj, nb_flag):
    """
    Modifies the nb flag for this object shares it.
    """
    if hasattr(obj,_REGISTER_NB_FLAG_ATTR):
      obj.nb_flag = nb_flag
      # print(f'in obj_register: {obj.nb_flag=}')

  @staticmethod
  def module_register(layer,nb_flag):
    """
    Modifies the new batch flag for all sub modules so the tensor is shared between them.

    After this set has been completed all modifying the persistent tensor "nb_flag" 
    will distribute the change to all other modules.
    """
    for l in NBFlag.modules(layer):
      # print('MODULE REGISTER 1', l.nb_flag, nb_flag)
      l.nb_flag = nb_flag  
      # print(f'---In module_register: {l.nb_flag=} {hex(id(l.nb_flag))=}---')

  @staticmethod
  def update(nb_flag, state):
    """
    Convenience function to change the state of flag

    nb_flag: A 0d tensor with one boolean item, allocateed by alloc_nb_flag
    state: True or False, depending  on the desired state of the flag
    """
    # print(f'IN UPDATE NB FLAG 1 {nb_flag=} {state=}')
    nb_flag.copy_(state)
    # print(f'in update: {nb_flag=}')
    # print(f'IN UPDATE NB FLAG 2 {nb_flag=} {hex(id(nb_flag))=} {state=}')

  @staticmethod
  def increment(nb_flag):
    """
    Convenience function to change the state of flag
    """
    # print(f'IN UPDATE NB FLAG 1 {nb_flag=} {state=}')
    nb_flag += 1
    # print(f'in update: {nb_flag=}')
    # print(f'IN UPDATE NB FLAG 2 {nb_flag=} {hex(id(nb_flag))=} {state=}')

  @staticmethod
  def modules(layer):
    """
    Assuming a "torch.nn.Module" build a generator for the for constructing the list
    of those with valid new batch flag registration.
    """
    return (l for l in layer.modules() if hasattr(l,_REGISTER_NB_FLAG_ATTR))

class NBFlagMixin:
  def register_nb(self, nb_flag):
    """
    Register a NB flag to change the behavior.
    """
    assert hasattr(super, _REGISTER_NB_FLAG_ATTR),\
           '"register_done" was called on an object (inheriting from "DoneFlagMixin")'\
           ' where no "nb_flag" attribute was created'
    self.nb_flag = nb_flag
