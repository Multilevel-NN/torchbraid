def main(de_vocab, en_vocab, device, args):
  # from dataloader import *

  import torch
  import torch.nn as nn
  from torch import Tensor
  from torch.nn import (TransformerEncoder, TransformerDecoder,
                        TransformerEncoderLayer, TransformerDecoderLayer)

  import math

  PAD_IDX_DE, PAD_IDX_EN = de_vocab['<pad>'], en_vocab['<pad>']

  class Seq2SeqTransformer(nn.Module):
      def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                   emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                   dim_feedforward:int = 512, dropout:float = 0.1):
          super(Seq2SeqTransformer, self).__init__()
          encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                  dim_feedforward=dim_feedforward)
          self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
          decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                  dim_feedforward=dim_feedforward)
          self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

          self.generator = nn.Linear(emb_size, tgt_vocab_size)
          self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
          self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
          self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

      def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                  tgt_mask: Tensor, src_padding_mask: Tensor,
                  tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
          src_emb = self.positional_encoding(self.src_tok_emb(src))
          tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
          memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
          outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                          tgt_padding_mask, memory_key_padding_mask)
          return self.generator(outs)

      def encode(self, src: Tensor, src_mask: Tensor):
          return self.transformer_encoder(self.positional_encoding(
                              self.src_tok_emb(src)), src_mask)

      def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
          return self.transformer_decoder(self.positional_encoding(
                            self.tgt_tok_emb(tgt)), memory,
                            tgt_mask)

  class PositionalEncoding(nn.Module):
      def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
          super(PositionalEncoding, self).__init__()
          den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
          pos = torch.arange(0, maxlen).reshape(maxlen, 1)
          pos_embedding = torch.zeros((maxlen, emb_size))
          pos_embedding[:, 0::2] = torch.sin(pos * den)
          pos_embedding[:, 1::2] = torch.cos(pos * den)
          pos_embedding = pos_embedding.unsqueeze(-2)

          self.dropout = nn.Dropout(dropout)
          self.register_buffer('pos_embedding', pos_embedding)

      def forward(self, token_embedding: Tensor):
          return self.dropout(token_embedding +
                              self.pos_embedding[:token_embedding.size(0),:])

  class TokenEmbedding(nn.Module):
      def __init__(self, vocab_size: int, emb_size):
          super(TokenEmbedding, self).__init__()
          self.embedding = nn.Embedding(vocab_size, emb_size)
          self.emb_size = emb_size
      def forward(self, tokens: Tensor):
          return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

  def generate_square_subsequent_mask(sz):
      mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
      mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
      return mask

  def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX_DE).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX_EN).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

  SRC_VOCAB_SIZE = len(de_vocab)
  TGT_VOCAB_SIZE = len(en_vocab)
  EMB_SIZE = 512
  NHEAD = 8
  FFN_HID_DIM = 512
  # BATCH_SIZE = 2#128
  NUM_ENCODER_LAYERS = args.n_lays_enc#3
  NUM_DECODER_LAYERS = args.n_lays_dec#3
  # NUM_EPOCHS = 16

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                   EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                   FFN_HID_DIM)

  for p in transformer.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)

  transformer = transformer.to(device)

  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX_EN)

  optimizer = torch.optim.Adam(
      transformer.parameters(), lr=args.lr,#0.0001, 
                                betas=(0.9, 0.98), eps=1e-9)

  return transformer, loss_fn, optimizer, create_mask, generate_square_subsequent_mask

if __name__ == '__main__':
  main()



























