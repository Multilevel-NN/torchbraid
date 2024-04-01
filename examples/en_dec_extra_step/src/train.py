import evaluate
import torch

try:
  metric = evaluate.load('sacrebleu')
except:
  print('sacrebleu failed. Loading bleu')
  metric = evaluate.load('bleu')

def train_epoch(_vars):
  print('Begin training')
  t0 = time.time()

  # Constants
  n_gradients_update = 10   # (for gradients accumulation)
  n_monitoring = 500*10
  n_early_stop = 10
  min_accuracy_early_stop = .95
  iter_valid_data_loader = iter(valid_data_loader)

  # Initialise variables
  counter_gradients_update = 0
  counter_monitoring = 0
  correct_training = 0 
  total_training = 0
  counter_accuracyValid_notBetter = 0
  max_accuracyValid = 0.
  losses_training = []
  losses_validation = []
  losses_temp_training = []
  accuracies_training = []
  accuracies_validation = []
  first_loss = True
  finished = False
  optimizer.zero_grad()

  ### Load model & history (only in algebra__linear_1d)
  '''
  try:
    checkpoint = torch.load(
      'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg1.pt'
    )
  except:
    checkpoint = torch.load(
      'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg2.pt'
    )
  model.load_state_dict(checkpoint['model_state'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  losses_training = checkpoint['losses_training']
  losses_validation = checkpoint['losses_validation']
  accuracies_training = checkpoint['accuracies_training']
  accuracies_validation = checkpoint['accuracies_validation']
  '''
  #####################################################

  while not finished:
    model.train()   # train
    for batch in train_data_loader:
      sources, targets = batch
      targets_input, targets_output = targets[:, :-1], targets[:, 1:]   # 6.2

      # Forward
      outputs = model(sources, targets_input)   
                                            # (batch, len_max_targets, alphabet)
      outputs_alphabet_dim1 = outputs.transpose(1, 2)   
                                            # (batch, alphabet, len_max_targets)
      loss = loss_function(outputs_alphabet_dim1, targets_output)
      predictions = outputs.argmax(dim=2)

      # Backward
      loss.backward()
      counter_gradients_update += 1
      if counter_gradients_update % n_gradients_update == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)  # Grad. clipping
        optimizer.step()    # 6.3 Gradient accumulation
        optimizer.zero_grad()

      correct_training += torch.logical_or(
          predictions == targets_output, 
          targets_output == vocabulary_target.pad_id
      ).prod(dim=1).sum(dim=0).item()
      total_training += targets_output.size()[0]

      # Monitoring
      losses_temp_training.append(loss.item())
      counter_monitoring += 1
      if first_loss:
        text = f'first training loss {loss.item() : .4e}'
        print(colored(text, 'blue') if colour_imported else text)
        print()
        first_loss = False

      if args.debug: print(loss.item())

      if counter_monitoring % n_monitoring == 0:
        model.eval()   # evaluate 
        with torch.no_grad():        
          # (I) training set
          losses_training.append(np.mean(losses_temp_training))
          accuracies_training.append(correct_training / total_training)
          correct_training, total_training = 0, 0

          text = f'training loss {loss.item() : .4e}'
          text += f'\ttraining accuracy {100 * accuracies_training[-1] : .4f}%'
          print(colored(text, 'blue') if colour_imported else text)

          # We'll take as an example the first entry in the batch
          print_example(predictions, sources, targets_output, vocabulary_source, 
                       vocabulary_target, 'blue')

          losses_temp_validation = []
          correct_validation, total_validation = 0, 0
          candidate_corpus, reference_corpus = [], []
          for _ in range(50):
            # (II) validation set
            batch = next(iter_valid_data_loader, None)
            if batch == None:
              iter_valid_data_loader = iter(valid_data_loader)
              batch = next(iter_valid_data_loader)

            sources, targets = batch
            targets_output = targets[:, 1:]

            # Forward
            # Greedy
            outputs, predictions = greedy(
                model, 
                sources, 
                targets.size()[1]-1, 
                vocabulary_target
            )   # outputs:     (batch, <eos>-ed length, alphabet)
                # predictions: (batch, <eos>-ed length)
            targets_output = targets_output[:, :predictions.size()[1]]
                                                        # (batch, <eos>-ed length)
            outputs_alphabet_dim1 = outputs.transpose(1, 2)   
                                              # (batch, alphabet, <eos>-ed length)
            loss = loss_function(outputs_alphabet_dim1, targets_output)
            losses_temp_validation.append(loss.item())

            correct_validation += torch.logical_or(
                predictions == targets_output, 
                targets_output == vocabulary_target.pad_id
            ).prod(dim=1).sum(dim=0).item()
            total_validation += targets_output.size()[0]

            for prediction, target_output in zip(predictions, targets_output):
              pred = ''.join([vocabulary_target.id_to_string[i.item()] \
                                                 for i in prediction])
              tgt = ''.join([vocabulary_target.id_to_string[i.item()] \
                                              for i in target_output])
              for char in '.,;:!?':
                pred = pred.replace(char, ' '+char)
                tgt = tgt.replace(char, ' '+char)
              pred = pred.replace('<eos>', ' <eos>')
              tgt = tgt.replace('<eos>', ' <eos>')

              candidate_corpus.append(pred.split())
              reference_corpus.append([tgt.split()])

          # 5. Accuracy computation
          # 5.1 (Also above in training set)
          accuracy_validation = correct_validation / total_validation

          # Monitoring
          # losses_validation.append(loss.item())
          losses_validation.append(np.mean(losses_temp_validation))
          accuracies_validation.append(accuracy_validation)

          text = f'validation loss {loss.item() : .4e}'
          text += f'\tvalidation accuracy {100*accuracies_validation[-1] :.4f}%'
          print(colored(text, 'red') if colour_imported else text)
          print_example(predictions, sources, targets_output, vocabulary_source, 
                       vocabulary_target, 'red')
          print(f'val bleu {bleu_score(candidate_corpus, reference_corpus)}')
          
          # Early stop: 
          if accuracy_validation > max_accuracyValid:
            max_accuracyValid = accuracy_validation
            counter_accuracyValid_notBetter = 0
          
          else:
            counter_accuracyValid_notBetter += 1
            condition1 = (counter_accuracyValid_notBetter >= n_early_stop)
                          # if valid. accuracy has not reached a global maximum 
                          #...within the last n_early_stop monitoring steps
            condition2 = ( min(
                accuracies_validation[-n_early_stop:]
            ) >= min_accuracy_early_stop )
                          # And all of the last n_early_stop monitoring steps 
                          #...are above min_accuracy_early_stop (only because
                          #...it is known that it will reach this accuracy)
            if condition1 and condition2:
              text = 'Early stopping: validation accuracy has not improved in '
              text += f'the last {n_early_stop} monitoring steps and the '
              text += 'minimum of these accuracies is above '
              text += f'{100 * min_accuracy_early_stop}%'
              print(text)
              finished = True   # Then, early stop. 
              break

          ### Save model & history (only in algebra__linear_1d)
          '''
          model_state = {
              'model_state': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'losses_training': losses_training,
              'losses_validation': losses_validation,
              'accuracies_training': accuracies_training,
              'accuracies_validation': accuracies_validation
          }
          torch.save(
            model_state, 
            'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg1.pt'
          )
          torch.save(
            model_state, 
            'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg2.pt'
          ) # save it in two files so that if it gets interrupted while saving
            #...one file (so the file becomes blank), the model can be recovered
            #...through the other file.
          '''
          #####################################################

      # Stop because time limit was exceeded.
      if time.time()-t0 >= time_limit_minutes * 60:
        print(f'Time limit of {time_limit_minutes} minutes has been exceeded.')
        finished = True
        break

def print_example(predictions, sources, targets_output, vocabulary_source, vocabulary_target, colour):
  question = sources[0,:]
  padding_question = (question == vocabulary_source.pad_id).nonzero()
  index_padding_question = padding_question[0] if len(
      padding_question != 0
  ) else len(question)
  text1 = ''.join(
      [
       vocabulary_source.id_to_string[i.item()] for i in question[
                                                      :index_padding_question]
      ]
  )

  answer_predicted = predictions[0,:]
  eos_answer_predicted = (
      answer_predicted == vocabulary_target.eos_id
  ).nonzero()
  index_eos_answer_predicted = eos_answer_predicted[0] if len(
      eos_answer_predicted != 0
  ) else len(answer_predicted)
  text2 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_predicted[
                                                :index_eos_answer_predicted+1]
      ]
  )

  answer_correct = targets_output[0,:]
  eos_answer_correct = (answer_correct == vocabulary_target.eos_id).nonzero()
  index_eos_answer_correct = eos_answer_correct[0] if len(
      eos_answer_correct != 0
  ) else len(answer_correct)
  text3 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_correct[
                                                  :index_eos_answer_correct+1]
      ]
  )
  text4 = bool(torch.logical_or(
      predictions == targets_output, 
      targets_output == vocabulary_target.pad_id
  ).prod(dim=1)[0].item())

  for pretext, text in [('QUESTION:', text1), 
                        ('PREDICTED ANSWER:', text2), 
                        ('CORRECT ANSWER:', text3), 
                        ('CORRECT?', 'Yes.' if text4 else 'No.')]:
    if colour_imported:
      print(colored(f'\t{pretext : >30} {text : <12}', colour))
    
    else:
      print(f'\t{pretext : >30} {text : <12}')

  print('\n\n')

def evaluate_bleu(_vars):
  _vars.model.eval() 
  with torch.no_grad():
    candidate_corpus, reference_corpus = [], []
    bleu = {'train': None, 'test': None}

    for mode in ['train', 'test']:
      for i, instance in enumerate(_vars.dl[mode]):
        print(i)
        # if _vars.debug and i > 2: break
        src = instance['input_ids']
        translation = instance['translation'][_vars.lang_tgt]
        outputs = _vars.model.generate(
          src,
          max_new_tokens=40, 
          do_sample=False,#True, 
          top_k=30, 
          top_p=0.95
        )
        for j, output in enumerate(outputs):
          candidate = _vars.tokenizer.decode(
            output, 
            skip_special_tokens=True
          )
          reference = [translation[j]]
          candidate_corpus.append(candidate)
          reference_corpus.append(reference)

      bleu[mode] = metric.compute(
        predictions=candidate_corpus, 
        references=reference_corpus,
      )

    _vars.candidate_corpus = candidate_corpus
    _vars.reference_corpus = reference_corpus
    _vars.bleu = bleu
    


































