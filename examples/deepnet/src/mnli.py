import datasets
import torch
from collections import OrderedDict
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from network_architecture_v2 import MyBertForSequenceClassification
import argparse
import logging
import numpy as np
logging.disable(logging.WARNING)

# Define a function to compute metrics
def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model on the MMNLI task")
    parser.add_argument('--bsize', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    # Load the MNLI dataset from the GLUE benchmark
    mnli_dataset = datasets.load_dataset('glue', 'mnli')

    # Load a pre-trained tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(
            examples['premise'], 
            examples['hypothesis'], 
            truncation='longest_first', 
            padding='max_length', 
            max_length=64,  # Adjust based on your model's max input length
            return_overflowing_tokens=False,  # Suppress the warning
        )

    print('Loading Dataset')
    encoded_dataset = mnli_dataset.map(preprocess_function, batched=True)

    # Load pre-trained model
    model_dicts = torch.load(f'bert-save-1/model_serial_checkpoint_batch_idx=80000')
    new_dict = OrderedDict(model_dicts['model_state_dict'])
    # Load actual model 

    model_serial = torch.load('serialnet_bert_32')
    model_serial.load_state_dict(new_dict)

    # Load a pre-trained model for sequence classification
    model = MyBertForSequenceClassification(model_serial, num_labels=3)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        evaluation_strategy='epoch',     # evaluate each epoch
        learning_rate=args.lr,              # learning rate
        per_device_train_batch_size=args.bsize,  # batch size for training
        per_device_eval_batch_size=args.bsize,   # batch size for evaluation
        num_train_epochs=5,              # number of training epochs
        weight_decay=0.1,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        warmup_ratio=.06,
        dataloader_drop_last=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=encoded_dataset['train'],         # training dataset
        eval_dataset=encoded_dataset['validation_matched'],     # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
