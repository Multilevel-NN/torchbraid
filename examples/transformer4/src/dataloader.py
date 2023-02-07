def main(train_data, val_data, test_data, de_vocab, en_vocab):
  # from data_processing import *

  from torch.nn.utils.rnn import pad_sequence
  from torch.utils.data import DataLoader

  BATCH_SIZE = 2#128
  PAD_IDX_DE, PAD_IDX_EN = de_vocab['<pad>'], en_vocab['<pad>']
  BOS_IDX_DE, BOS_IDX_EN = de_vocab['<sos>'], en_vocab['<sos>']
  EOS_IDX_DE, EOS_IDX_EN = de_vocab['<eos>'], en_vocab['<eos>']

  def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
      de_batch.append(torch.cat([torch.tensor([BOS_IDX_DE]), de_item, torch.tensor([EOS_IDX_DE])], dim=0))
      en_batch.append(torch.cat([torch.tensor([BOS_IDX_EN]), en_item, torch.tensor([EOS_IDX_EN])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX_DE)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX_EN)
    return de_batch, en_batch

  train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True)#, collate_fn=generate_batch)
  valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                          shuffle=True)#, collate_fn=generate_batch)
  test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=True)#, collate_fn=generate_batch)

  return train_iter, valid_iter, test_iter

if __name__ == '__main__': main()