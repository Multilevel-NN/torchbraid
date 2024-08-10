## Example and implementation inspired by: https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/utils/optimizers_and_distributions.py#L5
## Generation from https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L101

import torch
import torch.nn as nn

from autoinit import AutoInitializer

from generation          import generate
from load_save           import load, save
from positional_encoding import PositionalEncoding

@AutoInitializer
class Transformer(nn.Module):
  def __init__(
    self, d_model, nhead, num_encoder_layers, num_decoder_layers, 
    dim_feedforward, dropout, batch_first, norm_first, src_vocab, tgt_vocab, 
    device, initialize_parameters,
  ):
    super().__init__()

    self.src_embedding = nn.Embedding(len(src_vocab), d_model)
    self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model)

    self.positional_encoding = PositionalEncoding(
      d_model, dropout, batch_first,
    )

    encoder_layer = nn.TransformerEncoderLayer(
      d_model, nhead, dim_feedforward, dropout, batch_first=batch_first,
      norm_first=norm_first,
    )
    decoder_layer = nn.TransformerDecoderLayer(
      d_model, nhead, dim_feedforward, dropout, batch_first=batch_first,
      norm_first=norm_first,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    ## Try this:
    # self.encoder_final_ln = nn.LayerNorm(d_model)
    # self.decoder_final_ln = nn.LayerNorm(d_model)

    # print(f'self.encoder.use_nested_tensor: {self.encoder.use_nested_tensor}')

    self.lm_head = nn.Linear(d_model, len(tgt_vocab))
    self.log_softmax = nn.LogSoftmax(dim=-1)

    if initialize_parameters: self.initialize_parameters()
    self.to(device)

  def __repr__(self): 
    return ' '.join([
      f'Transformer model w/ d_model={self.d_model}, nhead={self.nhead},',
      f'num_encoder_layers={self.num_encoder_layers},',
      f'num_decoder_layers={self.num_decoder_layers},',
      f'dim_feedforward={self.dim_feedforward}, dropout={self.dropout},',
      f'batch_first={self.batch_first}, and norm_first={self.norm_first}.'
    ])

  ## Shape annotations: batch-first | not-batch-first
  def compute_logits(self, tgt, memory, decoder_masks):
    output_embeddings = self.compute_output_embeddings(tgt, memory, 
                                                       decoder_masks)
    logits = self.lm_head(output_embeddings)  #/ [b, Lt, m] #| [Ls, b, d]
    return logits if self.batch_first else logits.transpose(0, 1)

  def compute_memory(self, src, encoder_masks):  #/ src: [b, Ls]
    x = self.src_embedding(src if self.batch_first else src.T)  #/ [b, Ls, d] #| [Ls, b, d]
    x = self.positional_encoding(x)  #/ [b, Ls, d] #| [Ls, b, d]
    memory = self.encoder(x, **encoder_masks)  #/ [b, Ls, d] #| [Ls, b, d]
    return memory

  def compute_output_embeddings(self, tgt, memory, decoder_masks):
    y = self.tgt_embedding(tgt if self.batch_first else tgt.T)  #/ [b, Lt, d] #| [Lt, b, d]
    y = self.positional_encoding(y)  #/ [b, Lt, d] #| [Lt, b, d]
    output_embeddings = self.decoder(y, memory, **decoder_masks)  #/ [b, Lt, d] #| [Lt, b, d]
    return output_embeddings

  def initialize_parameters(self):
    ## Xavier
    for parameter in self.parameters():
        if parameter.ndim > 1: nn.init.xavier_uniform_(parameter)

  def forward(self, src, tgt):  #/ src: [b, Ls]
                                #/ tgt: [b, Lt]
    ## Encoder
    encoder_masks = self.get_encoder_masks(src)
    memory = self.compute_memory(src, encoder_masks)  #/ [b, Ls, d]

    ## Decoder
    decoder_masks = self.get_decoder_masks(tgt, encoder_masks)
    logits = self.compute_logits(tgt, memory, decoder_masks)  #/ [b, Lt, m]
    log_probs = self.log_softmax(logits)                      #/ [b, Lt, m]

    return log_probs

  def generate(self, *args, **kwargs):  return generate(self, *args, **kwargs)
    # self, src: torch.LongTensor, max_new_tokens: int, do_sample: bool, 
    # num_beams: int, num_return_sequences: int, top_k: int, top_p: float,

  def get_attention_mask(self, Lt):
    return nn.Transformer.generate_square_subsequent_mask(Lt).to(self.device)
    # return torch.triu(
    #   torch.ones((Lt, Lt), dtype=torch.bool, device=self.device),
    #   diagonal=1,
    # )

  def get_decoder_masks(self, tgt, encoder_masks):
    return {'tgt_mask': self.get_attention_mask(Lt=tgt.shape[1]),
            'tgt_key_padding_mask': 
              self.get_padding_mask(tgt, self.tgt_vocab.pad_id),
            'memory_key_padding_mask': encoder_masks['src_key_padding_mask']}

  def get_encoder_masks(self, src):
    return {'src_key_padding_mask': 
              self.get_padding_mask(src, self.src_vocab.pad_id)}

  def get_padding_mask(self, x, pad_id):
    return torch.zeros(x.shape, device=self.device) \
            .masked_fill((x == pad_id), float('-inf'))

  def load(self, *args, **kwargs): return load(self, *args, **kwargs)
  def save(self, *args, **kwargs): return save(self, *args, **kwargs)

def get_model(model_config, src_vocab, tgt_vocab, device): 
  return Transformer(
    **model_config.__dict__, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    device=device,
  )

if __name__ == '__main__':  # debugging
  class FakeVocab: 
    pad_id = 0
    def __len__(self): return 10

  d_model            = 8
  nhead              = 2
  num_encoder_layers = 6
  num_decoder_layers = 6
  dim_feedforward    = 32
  dropout            = 0.#.1
  # batch_first        = False
  # batch_first        = True
  src_vocab          = FakeVocab()
  tgt_vocab          = FakeVocab()
  device             = 'cpu'

  ## Dropout (PE + enc/dec) makes batch-first change the output

  T = []
  for batch_first in [False, True]:
    torch.manual_seed(0)

    model = Transformer(
      d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
      dropout, batch_first, src_vocab, tgt_vocab, device,
    )

    batch_size         = 5
    max_length_src     = 3
    max_length_tgt     = 4
    
    x = torch.randint(0, len(src_vocab), (batch_size, max_length_src))
    y = torch.randint(0, len(tgt_vocab), (batch_size, max_length_tgt))

    T.append(model(x, y))

  T1, T2 = T




