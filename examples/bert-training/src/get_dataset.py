from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
import torch

import random
from torch.utils.data import Dataset

# Modified from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
class MyBERTDataset(Dataset):
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
        # Assuming t1_random and t2_random are already PyTorch tensors
        CLS_token = torch.tensor([self.tokenizer.vocab['[CLS]']], dtype=t1_random.dtype, device=t1_random.device)
        SEP_token = torch.tensor([self.tokenizer.vocab['[SEP]']], dtype=t1_random.dtype, device=t1_random.device)
        PAD_token = torch.tensor([self.tokenizer.vocab['[PAD]']], dtype=t1_label.dtype, device=t1_label.device)

        # For t1 and t1_label
        t1 = torch.cat((CLS_token, t1_random, SEP_token))  # Insert [CLS] at the beginning and append [SEP] at the end
        t1_label = torch.cat((PAD_token, t1_label, PAD_token))  # Insert [PAD] at the beginning and append [PAD] at the end
        
        # For t2 and t2_label
        t2 = torch.cat((t2_random, SEP_token))  # Append [SEP] at the end
        t2_label = torch.cat((t2_label, PAD_token))  # Append [PAD] at the end

        # Step 4; combine into 1
        segment_label_t1 = torch.ones(len(t1), dtype=torch.long, device=t1.device)
        segment_label_t2 = torch.full((len(t2),), 2, dtype=torch.long, device=t2.device)  # Using 2 for the second segment
        segment_label = torch.cat((segment_label_t1, segment_label_t2))[:self.seq_len]
        
        # Concatenate t1 and t2 for bert_input and their labels
        bert_input = torch.cat((t1, t2))[:self.seq_len]
        bert_label = torch.cat((t1_label, t2_label))[:self.seq_len]
        
        # Padding
        PAD_token = self.tokenizer.vocab['[PAD]']
        if len(bert_input) < self.seq_len:
            padding_length = self.seq_len - len(bert_input)
            padding = torch.full((padding_length,), PAD_token, dtype=bert_input.dtype, device=bert_input.device)
            bert_input = torch.cat((bert_input, padding))
            bert_label = torch.cat((bert_label, padding))
            segment_label = torch.cat((segment_label, padding))
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return output

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

        ind_split = random.randrange(1, num_tokens - 1) 

        # These are the two sentences
        t1, t2 = tokens[0:ind_split], tokens[ind_split:]

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return torch.tensor(t1), torch.tensor(t2), 1
        else:
            # Need to grab random line and make it correct so that length is less than or equal to original t2
            rand_sentence = self.tokenized_data[random.randrange(len(self.tokenized_data))]['input_ids'][1:-1]
            if len(t2) >= len(rand_sentence):
                t2 = rand_sentence
            else:
                new_ind = random.randrange(0, len(rand_sentence) - len(t2))
                t2 = rand_sentence[new_ind:new_ind + len(t2)]

            return torch.tensor(t1), torch.tensor(t2), 0

    def random_word(self, sentence):
        # Assuming 'sentence' is a PyTorch tensor
        output_label = torch.zeros_like(sentence)
        output = sentence.clone()  # Create a copy of the input tensor for output
    
        # Calculate probabilities for each token in one go
        probs = torch.rand(sentence.size())
        mask_indices = (probs < 0.15).nonzero(as_tuple=True)[0]  # Indices where tokens will be modified
    
        # Calculate sub-probabilities for actions within the 15% chance
        action_probs = torch.rand(mask_indices.size(0))
    
        # 80% chance change token to mask token
        mask_tokens = mask_indices[action_probs < 0.8]
        output[mask_tokens] = self.tokenizer.vocab['[MASK]']
    
        # 10% chance change token to random token
        random_tokens = mask_indices[(action_probs >= 0.8) & (action_probs < 0.9)]
        if len(random_tokens) > 0:
            output[random_tokens] = torch.randint(1000, len(self.tokenizer.vocab), (len(random_tokens),))
    
        # For the 10% chance to keep the same token, no action is needed as we've copied the original tokens
    
        # Update output_label for changed tokens
        output_label[mask_indices] = sentence[mask_indices]
    
        return output, output_label        

def obtain_dataset(percent_data:float = 0.01, seq_len: int = 128):
    """
    See Jupyter for logic here
    """
    # Hard code for now
    if percent_data > 1:
        split = f'train[:{int(percent_data)}]'
    else:
        split = f'train[:{int(percent_data * 100)}%]'
    bookcorpus_train = load_dataset('bookcorpus', split=split, trust_remote_code=True)
    wiki_train = load_dataset("wikipedia", "20220301.simple", split=split, trust_remote_code=True)

    # bookcorpus_train = load_dataset('bookcorpus', split=f'train[0:25000]')
    # wiki_train = load_dataset("wikipedia", "20220301.simple", split=f'train[0:25000]')

    # bookcorpus_train = load_dataset('bookcorpus', split=f'train[0:100]')
    # wiki_train = load_dataset("wikipedia", "20220301.simple", split=f'train[0:100]')

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
    tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"], num_proc=8).filter(
        filter_short, num_proc=8
    )
    # print(tokenized_datasets)

    return MyBERTDataset(tokenized_datasets, tokenizer, seq_len), tokenizer.vocab_size
