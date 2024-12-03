from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
import torch
import spacy


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

        # Step 4: combine into one
        # Step 4a: if too long, truncate evenly first
        if len(t1_random) + len(t2_random) + 3 > self.seq_len:   # plus 3 for number of tokens additional
            total_length = len(t1_random) + len(t2_random) + 3 # Include 3 additional tokens
            excess_length = total_length - self.seq_len # This is the total number tokens which needs to be truncated

            # Truncate proportionally
            len_t1 = len(t1_random) - int(len(t1_random) / total_length * excess_length)
            len_t2 = len(t2_random) - int(len(t2_random) / total_length * excess_length)

            # Ensure the total length is exactly self.seq_len; a bit dumb and inefficient but it's okay
            while len_t1 + len_t2 + 3 > self.seq_len:
                if len_t1 > len_t2:
                    len_t1 -= 1
                else:
                    len_t2 -= 1            

            t1_random = t1_random[:len_t1]
            t2_random = t2_random[:len_t2]
            t1_label = t1_label[:len_t1]
            t2_label = t2_label[:len_t2]

            assert len(t1_random) + len(t2_random) + 3 == self.seq_len, f'{len(t1_random) + len(t2_random) + 3=} {len(t1_random) - len_t1} {len(t2_random) - len_t2}, {self.seq_len}'

        # For t1 and t1_label
        t1 = torch.cat((CLS_token, t1_random, SEP_token))  # Insert [CLS] at the beginning and append [SEP] at the end
        t1_label = torch.cat((PAD_token, t1_label, PAD_token))  # Insert [PAD] at the beginning and append [PAD] at the end
        
        # For t2 and t2_label
        t2 = torch.cat((t2_random, SEP_token))  # Append [SEP] at the end
        t2_label = torch.cat((t2_label, PAD_token))  # Append [PAD] at the end

        segment_label_t1 = torch.zeros(len(t1), dtype=torch.long, device=t1.device)
        segment_label_t2 = torch.ones((len(t2),), dtype=torch.long, device=t2.device)  # Using 2 for the second segment
        segment_label = torch.cat((segment_label_t1, segment_label_t2))

        attention_mask = torch.ones_like(segment_label)
        
        # Concatenate t1 and t2 for bert_input and their labels
        bert_input = torch.cat((t1, t2))
        bert_label = torch.cat((t1_label, t2_label))

        if len(bert_label) <= self.seq_len: # Need to pad
            padding_length = self.seq_len - len(bert_input)
            padding = torch.full((padding_length,), self.tokenizer.vocab['[PAD]'], dtype=bert_input.dtype, device=bert_input.device)
            
            bert_input = torch.cat((bert_input, padding))

            bert_label = torch.cat((bert_label, padding))

            segment_label = torch.cat((segment_label, torch.ones_like(padding)))
            
            attention_mask = torch.cat((attention_mask, padding))
            
        assert len(bert_input) == self.seq_len, f"{len(bert_input)=} {self.seq_len=} {bert_input=}"
        assert len(bert_label) == self.seq_len, f"{len(bert_label)=} {self.seq_len=} {bert_input=}"
        assert len(segment_label) == self.seq_len, f"{len(segment_label)=} {self.seq_len=} {bert_input=}"
        assert len(attention_mask) == self.seq_len, f"{len(attention_mask)=} {self.seq_len=} {bert_input=}"
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label, 
                  "attention_mask": attention_mask}

        return output

    def get_sent(self, index):
        """
        Grabs the two sentences from each data entry; remove the beginning and ending 
        """
        # Strip [CLS], [SEP] from each entry and truncate (- 3 so that we need to put back in CLS, and two SEPs)
        tokens = self.tokenized_data[index]['input_ids']
        ind_split = tokens.index(102)

        # These are the two sentences
        # t1 contains 101 start but no end, and t2 contains end but no start so need to parse out
        t1, t2 = tokens[1:ind_split], tokens[ind_split+1:-1] # Don't need the SEP to start the second sentence.
        
        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return torch.tensor(t1), torch.tensor(t2), 1
        else:
            # Grab a random second sentence
            rand_sentence = self.tokenized_data[random.randrange(len(self.tokenized_data))]['input_ids']
            ind_split = rand_sentence.index(102)
        
            return torch.tensor(t1), torch.tensor(rand_sentence[ind_split+1:-1]), 0

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

def preprocess_paragraphs(paragraphs, nlp):
    """Preprocess a batch of paragraphs."""
    sentence_pairs_batch = []
    # for paragraph in paragraphs:
    docs = nlp.pipe(paragraphs, batch_size=128)
    for doc in docs:
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences) >= 2:
            # There might be an issue where half the data is overly related; e.g. (sentence 1, sentence 2), (sentence 2, sentence 3) sentence 2 appears twice! 
            sentence_pairs = [(sentences[i], sentences[i+1]) for i in range(len(sentences) - 1) if len(sentences[i]) > 1 and len(sentences[i + 1]) > 1] 
            sentence_pairs_batch.extend(sentence_pairs)   
    return sentence_pairs_batch

def preprocess_articles(batch, nlp):
    """Preprocess a batch of articles."""
    sentence_pairs_batch = []
    
    for article_text in batch['text']:
        # For sake of number of articles
        paragraphs = article_text.split('\n')

        # Don't want empty lines; also want substantial paragraphs and don't want headers; not particularly interested in accuracy of final model so this is fine. 
        paragraphs = [para for para in paragraphs if len(para.strip()) > 100] 

        # Preprocess only first few paragraphs
        sentence_pairs = preprocess_paragraphs(paragraphs[0:5], nlp)
        sentence_pairs_batch.extend(sentence_pairs)
            
    return {"sentence_pairs": sentence_pairs_batch}

def obtain_dataset(percent_data:float = 0.01, seq_len: int = 128):
    """
    Fairly simple way to pre-process wikipedia data. We throw away a lot of data, but basically for each article, we choose 
    several sentences using spaCy to be able to accurately do NSP. 
    """
    # Hard code for now
    if percent_data > 1:
        split = f'train[:{int(percent_data)}]'
    else:
        split = f'train[:{int(percent_data * 100)}%]'
    try:
        # bookcorpus_train = load_dataset('bookcorpus', split=split, trust_remote_code=True)
        wiki_train = load_dataset("wikipedia", "20220301.en", split=split, trust_remote_code=True)
    except:
        # bookcorpus_train = load_dataset('bookcorpus', split=split)
        wiki_train = load_dataset("wikipedia", "20220301.en", split=split)
 
    wiki_train = wiki_train.remove_columns([col for col in wiki_train.column_names if col != "text"]) # Only keep text
    # assert bookcorpus_train.features.type == wiki_train.features.type
    # raw_datasets = concatenate_datasets([bookcorpus_train, wiki_train])

    # We process the dataset to obtain sentence pairs
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    print(f'Using GPU with spacy: {spacy.prefer_gpu()}')
    def my_map(batch):
        result = preprocess_articles(batch, nlp)
        return result
        
    processed_dataset = wiki_train.map(my_map, batched=True, remove_columns=wiki_train.column_names, batch_size=64, load_from_cache_file=False)

    # Load pretrained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def group_texts(examples):
        tokenized_inputs = tokenizer(
            examples["sentence_pairs"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
        )
        return tokenized_inputs
    
    # preprocess dataset
    tokenized_datasets = processed_dataset.map(group_texts, batched=True,  num_proc=8, load_from_cache_file=False) # remove_columns=["text"],
        
    return MyBERTDataset(tokenized_datasets, tokenizer, seq_len), tokenizer.vocab_size

if __name__ == "__main__":
    myds, _ = obtain_dataset(100)

    print('loaded dataset')
    print(myds[99])
    print()
    # print(myds[0])
    # print(myds[3])
