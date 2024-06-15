import numpy as np
import torch
import torch.nn as nn

from ..model_utils.positional_encoding import PositionalEncoding

class PreContinuousBlock(nn.Module):
  def __init__(
    self, source_vocabulary, target_vocabulary, model_dimension, device, 
    **kwargs,
  ):
    super().__init__()

    ## Constants
    dim_alphabet_source = len(source_vocabulary)
    dim_alphabet_target = len(target_vocabulary)
    self.source_vocabulary = source_vocabulary
    self.target_vocabulary = target_vocabulary
    self.model_dimension = model_dimension
    self.device = device

    ## Embedding & Positional encoding
    self.embedding_encoder = nn.Embedding(
      num_embeddings=dim_alphabet_source, 
      embedding_dim=model_dimension,
      # padding_idx=source_vocabulary.pad_id,
    )
    self.embedding_decoder = nn.Embedding(
      num_embeddings=dim_alphabet_target, 
      embedding_dim=model_dimension,
      # padding_idx=target_vocabulary.pad_id,
    )
    self.positional_encoder = PositionalEncoding(model_dimension)

  def forward(self, **state):
    state_update = {}
    state_update.update(self.embed_src(**state))
    state_update.update(self.embed_tgt(**state))
    return state_update

  def embed_src(self, x, **kwargs):  # x: [b, L   ]
    src = x  # src: [b, L ]

    ## Padding masks for attention
    # src_padding_mask = torch.where(src.eq(self.pad_token_id), -np.inf, 0)  # src_padding_mask: [b, L]
    src_padding_mask = (src == self.source_vocabulary.pad_id)  # src_padding_mask: [b, L]
    mem_padding_mask = src_padding_mask                        # mem_padding_mask: [b, L]

    src = src.transpose(0, 1)   # (L, b)

    ## Embedding
    x = self.embedding_encoder(src)  # src: [L, b, d]

    ## Scaling
    # x *= np.sqrt(self.model_dimension)

    ## Positional encoding
    x = self.positional_encoder(x)  # x: [L, b, d]

    return {
      'x': x, 
      'src_padding_mask': src_padding_mask, 
      'mem_padding_mask': mem_padding_mask,
    }

  def embed_tgt(self, y, split_target=True, **kwargs):  # y: [b, L'+1]
    '''split_target is True during a conventional forward pass, where the 
    target must be split into target_inputs (to the model) and labels.
    However, during generation, the targe_inputs are the whole target tensor,
    so split_arget is False.'''

    if split_target: 
      tgt = y[:, :-1]    #    tgt: [b, L']
      labels = y[:, 1:]  # labels: [b, L']
    else:
      tgt = y
      labels = None

    ## Causal mask for attention
    Lp = tgt.shape[1]
    tgt_attention_mask = nn.Transformer.generate_square_subsequent_mask(sz=Lp) \
                         .to(self.device)    # (Lp, Lp)

    ## Padding mask for attention
    # tgt_padding_mask = torch.where(tgt.eq(self.pad_token_id), -np.inf, 0)  # tgt_padding_mask: [b, L']
    tgt_padding_mask = (tgt == self.target_vocabulary.pad_id)  # tgt_padding_mask: [b, L']

    tgt = tgt.transpose(0, 1)   # (L', b)

    ## Embedding
    y = self.embedding_decoder(tgt)  # tgt: [L', b, d]

    ## Scaling
    # y *= np.sqrt(self.model_dimension)

    ## Positional encoding
    y = self.positional_encoder(y)  # y: [L', b, d]

    return {
      'y': y,
      'tgt_attention_mask': tgt_attention_mask,
      'tgt_padding_mask': tgt_padding_mask,
      'target': labels,
    }




