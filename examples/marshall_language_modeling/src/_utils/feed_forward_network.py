import torch.nn as nn
from torchbraid.utils import LPDropout

class FeedForward(nn.Module):
  def __init__(self, model_dimension, dropout):
    super().__init__()
    self.net = nn.Sequential(
      #nn.Linear(model_dimension, 1, bias=False),
      nn.Linear(model_dimension, model_dimension, bias=True),

      #nn.Linear(model_dimension, 1 * model_dimension, bias=False),
      #nn.Linear(2, model_dimension, bias=False),
      #nn.Linear(model_dimension, 1),
      nn.GELU(),
      #nn.Linear(1, model_dimension, bias=False),
      #nn.Linear(1 * model_dimension, model_dimension, bias=False),
      #LPDropout(dropout),
    )

    #self.lin1 = nn.Linear(model_dimension, model_dimension, bias=False)
    #self.lin2 = nn.Linear(model_dimension, model_dimension, bias=False)
    
  def forward(self, x):
    return self.net(x)
    #return self.lin1(x)
    #x = self.lin1(x)
    #x = self.lin2(x)
    #return x
