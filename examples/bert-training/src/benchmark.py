import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

from network_architecture_v2 import MyBertForSequenceClassification

trained_model = torch.load(f'serial_net_hf_bert_12_epoch=5')

from get_dataset import obtain_dataset
from torch.utils.data import DataLoader

# This is just to grab and test a data
ds, vocab_size = obtain_dataset(percent_data = 500, seq_len=64)
train_size, test_size = int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)  # 80/20 split by default
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
print(f'{vocab_size=}')
train_loader = DataLoader(
train_ds, batch_size=32, shuffle=False, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
test_ds, batch_size=32, shuffle=False, pin_memory=True, drop_last=True
)

batch_iterator = iter(train_loader)  # Create an iterator from the DataLoader
single_batch = next(batch_iterator)   # Get a single batch

model = MyBertForSequenceClassification(trained_model)
model.to('cuda')
model.train()

bert_input = single_batch['bert_input'].to('cuda')
model(bert_input)

dataset = load_dataset("glue", "sst2")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Load the accuracy metric
metric = load_metric("accuracy")

# Define the compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)