# from training import *

import torch

def greedy_decode(model, src, src_mask, max_len, start_symbol, EOS_IDX_TGT, device, generate_square_subsequent_mask):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX_TGT:
          break
    return ys


def translate(model, src, src_vocab, tgt_vocab, device, generate_square_subsequent_mask):
  SOS_IDX_SRC, SOS_IDX_TGT = src_vocab['<sos>'], tgt_vocab['<sos>']
  EOS_IDX_SRC, EOS_IDX_TGT = src_vocab['<eos>'], tgt_vocab['<eos>']
  UNK_IDX_SRC = src_vocab['<unk>']

  tgt_vocab_inv = {v: k for (k, v) in tgt_vocab.items()}

  model.eval()
  tokens = [SOS_IDX_SRC] + [src_vocab.get(tok, UNK_IDX_SRC) for tok in src.strip().split()]+ [EOS_IDX_SRC]
  num_tokens = len(tokens)
  src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX_TGT, 
    EOS_IDX_TGT=EOS_IDX_TGT, device=device, generate_square_subsequent_mask=generate_square_subsequent_mask).flatten()
  # print(src, tgt_tokens)
  return " ".join([tgt_vocab_inv[tok.item()] for tok in tgt_tokens]).replace("<sos>", "").replace("<eos>", "")




























