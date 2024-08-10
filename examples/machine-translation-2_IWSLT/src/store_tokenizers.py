## INFO: This script has run via ipython on the cluster due to problems to import de_core_news_sm and en_core_web_sm (or, equivalently, spacy.load(...)) when running the .job file.

import de_core_news_sm
import en_core_web_sm
import os
import pickle

# OUTPUT_DIR = '../tokenizers'
OUTPUT_DIR = '/Users/marcsalvado/Desktop/Aux-Scripts-python/85_cheap-Transformer-IWSLT/tokenizers'

de_tokenizer = de_core_news_sm.load().tokenizer
en_tokenizer = en_core_web_sm.load().tokenizer

with open(os.path.join(OUTPUT_DIR, 'de_tokenizer.pkl'), 'wb') as f: 
  pickle.dump(de_tokenizer, f)
with open(os.path.join(OUTPUT_DIR, 'en_tokenizer.pkl'), 'wb') as f: 
  pickle.dump(en_tokenizer, f)
