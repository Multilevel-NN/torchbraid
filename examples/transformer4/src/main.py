## Example of de-en translation taken from PyTorch website: https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html

# from data_processing import *
# from dataloader import *
# from training import *
# from transformer import *
from translating import *

def main():
  for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    end_time = time.time()
    val_loss = evaluate(transformer, valid_iter)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"))

  translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu .", 
                                          de_vocab, en_vocab, de_tokenizer)

if __name__ == '__main__':
  main()
