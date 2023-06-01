import torch

def greedy_decode(model, src, max_len, start_symbol, EOS_IDX_TGT, device, generate_square_subsequent_mask):
    memory = model.encode(src)
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

def translate(model, prompt, voc, dev):
  voc_src, voc_tgt = voc['de'], voc['en']
  model.eval()
  src = [voc_src['<sos>']] + [voc_src[tok] for tok in prompt.strip().split()] \
                                                         + [voc_src['<eos>']]
  src = torch.tensor(src, dtype=torch.long).unsqueeze(dim=0).to(dev)
  tgt_tokens = greedy_decode(model,  src, src_mask, num_tokens+5, voc, dev)#.flatten()
  print(tgt_tokens.shape)
  return " ".join([tgt_vocab_inv[tok.item()] for tok in tgt_tokens]).replace("<sos>", "").replace("<eos>", "")


























