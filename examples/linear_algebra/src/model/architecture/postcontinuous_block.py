import torch.nn as nn

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, target_vocabulary, **kwargs):
    super().__init__()

    dim_alphabet_target = len(target_vocabulary)

    ## Language Modeling head
    self.LM_head = nn.Linear(
      model_dimension, dim_alphabet_target, bias=True,
    )

    # self.apply(init_weights)

  def forward(self, y, **kwargs):  # y: [L', b, d]
    y = y.transpose(0, 1)  # y: [b, L', d]
    logits = self.LM_head(input=y)  # logits: [b, L', m]

    return {'x': logits}


