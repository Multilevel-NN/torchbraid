import torch.nn as nn
from torchbraid.utils import LPDropout

class FeedForward(nn.Module):
  def __init__(self, model_dimension, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(model_dimension, 4 * model_dimension),
      nn.ReLU(),
      nn.Linear(4 * model_dimension, model_dimension),
      LPDropout(dropout),
    )

  def forward(self, x):
    return self.net(x)
