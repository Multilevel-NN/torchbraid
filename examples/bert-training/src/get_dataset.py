from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
import torch

import random
from torch.utils.data import Dataset

# Modified from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
class BERTDataset(Dataset):
    """
    Construct a BERT Dataset by 

    1. Doing next sentence prediction by taking a random integer around half the seq_len
    2. Doing the masked language model 
    """
    def __init__(self, tokenized_data: Dataset, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.tokenized_data = tokenized_data
        self.seq_len = seq_len
        
    def __len__(self):
        """
        Each tokenized data should be a diff sample
        """
        return len(self.tokenized_data)

    def __getitem__(self, item):
        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: modify and replace random word with mask / random tokens
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: now put it all together with CLS, SEP and finish with PAD
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4; combine into 1
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}
    def get_sent(self, index):
        """
        Grab random sentence pair by splitting the tokens randomly. This only samples the first part, but it's okay. 

        Randomly generates split, then randomly gives correct second or incorrect second sentence
        """
        
        # Strip [CLS], [SEP] from each entry and truncate (- 3 so that we need to put back in CLS, and two SEPs)
        tokens = self.tokenized_data[index]['input_ids'][1:-1]
        num_tokens = len(tokens)
        if num_tokens > self.seq_len - 3:
            tokens = tokens[0:self.seq_len - 3]
            num_tokens = len(tokens)
        # print(index, tokens, self.tokenizer.decode(tokens))

        ind_split = random.randrange(1, num_tokens - 1) 

        # These are the two sentences
        t1, t2 = tokens[0:ind_split], tokens[ind_split:]

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            # Need to grab random line and make it correct so that length is less than or equal to original t2
            rand_sentence = self.tokenized_data[random.randrange(len(self.tokenized_data))]['input_ids'][1:-1]
            if len(t2) >= len(rand_sentence):
                t2 = rand_sentence
            else:
                new_ind = random.randrange(0, len(rand_sentence) - len(t2))
                t2 = rand_sentence[new_ind:new_ind + len(t2)]

            return t1, t2, 0

    def random_word(self, sentence):
        output_label = []
        output = []

        for i, token_id in enumerate(sentence):
            prob = random.random()
            
            # 15% of the tokens would be replaced
            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token (don't want to give it bad tokens, start from 1000 which is where BERT Tokenizer has good tokens
                elif prob < 0.9:
                    output.append(random.randrange(1000, len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                output_label.append(0)

        # flattening (don't have to do this)
        # output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        # output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)

        return output, output_label
        

def obtain_dataset(percent_data:float = 0.01, seq_len: int = 128):
    """
    See Jupyter for logic here
    """
    # Hard code for now
    # bookcorpus_train = load_dataset('bookcorpus', split=f'train[:{int(percent_data * 100)}%]')
    # wiki_train = load_dataset("wikipedia", "20220301.simple", split=f'train[:{int(percent_data * 100)}%]')

    bookcorpus_train = load_dataset('bookcorpus', split=f'train[0:50000]')
    wiki_train = load_dataset("wikipedia", "20220301.simple", split=f'train[0:50000]')

    wiki_train = wiki_train.remove_columns([col for col in wiki_train.column_names if col != "text"]) # Only keep text
    assert bookcorpus_train.features.type == wiki_train.features.type
    raw_datasets = concatenate_datasets([bookcorpus_train, wiki_train])

    # Load pretrained 
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def group_texts(examples):
        tokenized_inputs = tokenizer(
            examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
        )
        
        return tokenized_inputs
    
    def filter_short(examples):
        return len(examples['input_ids']) > 6

    # preprocess dataset
    tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"]).filter(
        filter_short
    )
    # print(tokenized_datasets)

    return BERTDataset(tokenized_datasets, tokenizer, seq_len), tokenizer.vocab_size

if __name__ == "__main__":
    print("Run with main please.")
