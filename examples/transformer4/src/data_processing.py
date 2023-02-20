## Example of de-en translation
## Dataset taken from PyTorch website: https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html

import torch
from torchtext.utils import download_from_url, extract_archive

class Voc:
  def __init__(self):
    self.specials = tuple(f'<{c}>' for c in 'pad unk sos eos'.split())
    self.str2id = dict(zip(self.specials, range(len(self.specials))))
    self.id2str = {v: k for (k,v) in self.str2id.items()}

  def __getitem__(self, q):
    assert type(q) in [str, int]
    return self.str2id.get(q, self.str2id['<unk>']) \
              if type(q) == str else self.id2str[q]

  def __len__(self): return len(self.str2id)

  def extend(self, s):
    self.str2id[s] = len(self.str2id)
    self.id2str[len(self.id2str)] = s

def build_ds(fn, voc, max_len, dev, extend):
  ds = []
  langs = ['de', 'en']
  f = {lang: open(fn[lang], 'r') for lang in langs}
  while True:
    line, tensor = [{lang: None for lang in langs} for _ in range(2)]
    try: 
      for lang in langs: line[lang] = next(f[lang]).strip()
    except: break
    for lang in langs:
      line[lang] = line[lang].lower()
      for c in ['.', ',', ';', ':', '?', '!']:
        line[lang] = line[lang].replace(c, f' {c}')
      sent = []
      for tok in line[lang].split():
        if tok not in voc[lang].str2id and extend: voc[lang].extend(tok)
        sent.append(voc[lang].str2id.get(tok, voc[lang].str2id['<unk>']))
      for _ in range(max_len - len(sent)): sent.append(voc[lang].str2id['<pad>'])  # padding
      tensor[lang] = torch.tensor(sent, dtype=torch.long, device=dev)
    ds.append((tensor['de'], tensor['en']))
  return ds

def main(dev, max_len, batch_size):
  torch.manual_seed(0)
  # torch.use_deterministic_algorithms(True)

  url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
  url_tr = ('train.de.gz', 'train.en.gz')
  url_va = ('val.de.gz', 'val.en.gz')
  url_te = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

  fn_tr = dict(zip('de en'.split(), [extract_archive(download_from_url(url_base + url))[0] for url in url_tr]))
  fn_va = dict(zip('de en'.split(), [extract_archive(download_from_url(url_base + url))[0] for url in url_va]))
  fn_te = dict(zip('de en'.split(), [extract_archive(download_from_url(url_base + url))[0] for url in url_te]))

  ## Build ds
  voc = {'de': Voc(), 'en': Voc()}
  ds = {}
  ds['tr'] = build_ds(fn_tr, voc, max_len, dev, True)
  ds['va'] = build_ds(fn_va, voc, max_len, dev, False)
  ds['te'] = build_ds(fn_te, voc, max_len, dev, False)

  dl = {mod: torch.utils.data.DataLoader(ds[mod], batch_size=batch_size, 
    shuffle=True, drop_last=False) for mod in ('tr va te'.split())}

  return ds, dl, voc

if __name__ == '__main__':
  main()











