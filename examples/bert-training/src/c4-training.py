from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments, BertConfig, DataCollatorForLanguageModeling,  BertTokenizer
import torch
import torch.nn.functional as F
from datetime import datetime

# Load C4 Dataset
print('Loading dataset')
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
dataset = dataset.remove_columns(['timestamp', 'url'])

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.with_format("torch")

# Collator; can do MLM with higher prob
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.20) # Doing higher 

# Initialize the BERT configuration and model for Masked Language Modeling from scratch
config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=30, # Changed to 32
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.05,
        attention_probs_dropout_prob=0.05,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.01,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
)
model = BertForMaskedLM(config)


date = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print('Model loaded; training')
# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./bert-c4-results-32-{date}",
    #overwrite_output_dir=True,
    #save_steps=5,
    save_steps=100,
    save_strategy="steps",
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    per_device_train_batch_size=256,
    max_steps=10000 * 32,
    warmup_ratio=0.08, 
    save_total_limit=200,
    prediction_loss_only=True,
    dataloader_drop_last=True,
    max_grad_norm=0,
    lr_scheduler_type='linear',
    logging_first_step=True,
    logging_strategy='steps',
    logging_dir="./logs-32",  # Directory for storing logs
    logging_steps=8,      
    log_level='debug',       # Set log level to info
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
) 

from transformers import TrainerCallback
# Custom callback to print loss
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Step: {state.global_step}, Loss: {logs['loss']}")
trainer.add_callback(CustomCallback())

trainer.train()

# print(trainer.state.log_history)

