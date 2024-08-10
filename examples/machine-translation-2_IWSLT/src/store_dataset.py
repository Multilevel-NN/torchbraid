## INFO: This script has run locally and the resulting dataset been sent to the cluster due to problems to download the dataset on the cluster.

import os
from   tqdm import tqdm

from data import download_dataset, DOWNLODED_DATASET_FNM_ as DATASET_FNM_, SEP

DATASET_DIR = os.path.join('..', 'data')

def create_offline_dataset():
  dataset = download_dataset()
  splits = list(dataset.keys())

  for split in splits:
    de_dataset_path = os.path.join(DATASET_DIR, DATASET_FNM_('de', split))
    en_dataset_path = os.path.join(DATASET_DIR, DATASET_FNM_('en', split))

    with open(de_dataset_path, 'w') as de_f, \
         open(en_dataset_path, 'w') as en_f:

      for entry in tqdm(dataset[split]):
        de_sentence = entry['translation']['de']
        en_sentence = entry['translation']['en']

        de_f.write(de_sentence + SEP)
        en_f.write(en_sentence + SEP)

def main(): create_offline_dataset()

if __name__ == '__main__':
  main()
