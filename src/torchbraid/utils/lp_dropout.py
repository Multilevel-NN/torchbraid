"""
Dropout layer for LP tasks

Note that the classical nn.Dropout of PyTorch returns a new mask at every call, meaning that 
convergence might be an issue at high probability that the multigrid will not converge. We 
introduce a new flag `NBFlag` (New Batch Flag) which tells us when a new batch is happening 
and to introduce a new mask only then. However, due to some technical issues, we're currently
seeding the masks with this flag term, meaning that on each batch, the dropout is the same neurons. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbraid.utils import *


__all__ = ['LPDropout']

class LPDropout(NBFlagMixin, nn.Module):
    __constants__ = ['p', 'inplace']
    def __init__(self, p: float=0.2):
        """
        Same as regular Dropout, however, will only update on each new training call and not every call.
        Note that the seed is set for now to the flag value, meaning that every batch is the same dropout 
        basically
        
        Below is copied pasted documentation from PyTorch:

        During training, randomly zeroes some of the elements of the input
        tensor with probability :attr:`p` using samples from a Bernoulli
        distribution. Each channel will be zeroed out independently on every forward
        call.

        This has proven to be an effective technique for regularization and
        preventing the co-adaptation of neurons as described in the paper
        `Improving neural networks by preventing co-adaptation of feature
        detectors`_ .

        Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
        training. This means that during evaluation the module simply computes an
        identity function.

        Args:
            p: probability of an element to be zeroed. Default: 0.2

        Shape:
            - Input: :math:`(*)`. Input can be of any shape
            - Output: :math:`(*)`. Output is of the same shape as input
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

        self.register_buffer("p", torch.tensor(p))

        self.nb_flag = NBFlag.allocate()

        # This seems like it causes issues, when .to(device) is called; we don't want two flags around
        # self.register_buffer("nb_flag", nb_flag) 

        self.int_counter = torch.tensor(1)
        # self.register_buffer("int_counter", int_counter)

        self.mask = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If not training; it's just identity
        if not self.training:
            return input
        
        # If no self.mask, need to create one for initial batch
        if self.mask is None: 
            torch.manual_seed(0)

            # print(f'\t Generating mask; init could be due to dry run')

            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            self.mask = binomial.sample(input.size()) * (1.0/(1-self.p)) 

            # print(f'\t Generating mask; init could be due to dry run {self.mask=}')
            
        # Else if counter-mismatch, only happen if new batch 
        if self.int_counter != self.nb_flag:
            self.int_counter = self.nb_flag.detach().clone()

            torch.manual_seed(self.int_counter)
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            self.mask = binomial.sample(input.size()) * (1.0/(1-self.p))
            # print(f'\tGenerating mask:{self.nb_flag=} {self.int_counter=} {hex(id(self))=} {self.mask=}')

            
        return input * self.mask

    def extra_repr(self) -> str:
        return f'p={self.p}'