import torch
import torch.nn.functional as F
import torchmetrics.text
from torch.optim.lr_scheduler import  CosineAnnealingLR, LinearLR, SequentialLR
import jiwer
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random
import sys
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def create_dirs():
  if not os.path.exists('result/'):
    os.makedirs('result/')

  if not os.path.exists('result/config'):
    os.makedirs('result/config')

  if not os.path.exists('result/model/'):
    os.makedirs('result/model/')

  if not os.path.exists('result/train/'):
    os.makedirs('result/train/')

def create_files():
  if os.path.exists('result/config/hyperparameters.txt')==True:
    None
  else:
    model_params = dict(d_model=320, num_encoder_layer=2, num_decoder_layer=3, nhead=4, dim_forward=640, dropout=0.1,
               vocab_size=1, planes = [8, 16, 32, 64])
    with open('result/config/hyperparameters.txt', 'w') as f:
      f.write(str(model_params))

  if os.path.exists('result/config/augment.txt')==True:
    None
  else:
    augment_params = dict(add_noise_prob=0.2, noise_level=0.01, time_mask_param=10, time_mask_prob = 0.1,
                          freq_mask_param=10, freq_mask_prob = 0.1, gain_factor=[0.8, 1.2], gain_prob = 0.1,
                          clip_threshold=0.2, clipping_distortion_prob = 0.1)
    with open('result/config/augment.txt', 'w') as f:
      f.write(str(augment_params))
def create_dirs_test(split):

  if not os.path.exists('result/'):
    os.makedirs('result/')

  if not os.path.exists(f'result/{split}'):
    os.makedirs(f'result/{split}')

def print_details(model_num_params, model_config, num_train_audio, num_valid_audio, vocab_size, use_augment, augment_params):
  print("Model:")
  print("num trainable parameters:       ", model_num_params, ' Million')
  print(f"d_model = {model_config['d_model']}      num_encoder_layer = {model_config['num_encoder_layer']}     num_decoder_layer = {model_config['num_decoder_layer']}\n"
        f"nhead = {model_config['nhead']}         dim_forward = {model_config['dim_forward']}                 dropout = {model_config['dropout']}\n"
        f"vocab_size = {model_config['vocab_size']}    planes = {model_config['planes']}")
  print('------------------------------------------------------------------------------')
  print("Dataset:")
  print(f"train have {num_train_audio} audios")
  print(f"valid have {num_valid_audio} audios")
  print(f"vocab size = {vocab_size}")
  print('------------------------------------------------------------------------------')
  print("Augmentation:")
  if use_augment == 'True':
    print(f"add_noise_prob  = {augment_params['add_noise_prob']}  noise_level = {augment_params['noise_level']}\n"
          f"time_mask_param = {augment_params['time_mask_param']}  time_mask_prob = {augment_params['time_mask_prob']}\n"
          f"freq_mask_param = {augment_params['freq_mask_param']}  freq_mask_prob = {augment_params['freq_mask_prob']}\n"
          f"gain_factor     = {augment_params['gain_factor']}      gain_prob = {augment_params['gain_prob']}\n"
          f"clip_threshold  = {augment_params['clip_threshold']}   clipping_distortion_prob = {augment_params['clipping_distortion_prob']}")
  else:
    print("No Augmentation")

def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return device

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

def get_changes(con, model, device, num_epochs, learning_rate):
  if con == "True":
    model.load_state_dict(torch.load("result/model/last_model.pt", weights_only=True, map_location=device))
    lr_changes, wait, \
    best_wer, best_epoch, \
    train_loss_hist, \
    train_wer_hist, \
    valid_loss_hist, \
    valid_wer_hist, w_op_sch = torch.load("result/train/info.pt", weights_only=True, map_location=device)
    start, stop = len(train_loss_hist), len(train_loss_hist) + num_epochs
    lr = lr_changes[-1]
  else:
    train_loss_hist = []
    train_wer_hist = []
    valid_loss_hist = []
    valid_wer_hist = []
    best_wer = torch.inf
    best_epoch = 0
    wait = 0
    start, stop = 0, num_epochs
    lr = learning_rate
    lr_changes = []
    w_op_sch = None

  return train_loss_hist, train_wer_hist, valid_loss_hist, valid_wer_hist, best_wer, best_epoch, wait, start, stop, lr, lr_changes, w_op_sch


def to_sentence(preds, inputs, vocab):
  vocab_list = vocab.get_itos()
  def indexes_to_sentence(indexes):
    sentence = ''.join([vocab_list[index] for index in indexes])
    return sentence.replace('>', '').replace('<', '').replace('=', '')

  sentences_preds = []
  for pred in preds:
    if len(pred.shape) == 2:
      indexes = pred.argmax(dim=-1).tolist()
    else:
      indexes = pred.tolist()
    sentences_preds.append(indexes_to_sentence(indexes))

  sentences_inputs = [indexes_to_sentence(input.tolist()) for input in inputs]

  return sentences_preds, sentences_inputs

def smooth(input):
    sigma = 7
    smoothed = gaussian_filter1d(input, sigma)
    return smoothed
def plot_and_save_result(info):
  lr_changes, _,   \
  _, _,            \
  train_loss_hist, \
  train_wer_hist, \
  valid_loss_hist, \
  valid_wer_hist, _ = info
  plt.figure(figsize = (12, 8))
  x = range(0, len(train_wer_hist))
  plt.subplot(1, 2, 1)
  plt.plot(x, train_loss_hist, label = "Train Loss", color = "blue")
  plt.plot(x, valid_loss_hist, label = "Valid Loss", color = "red")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.subplot(1, 2, 2)
  plt.plot(x, train_wer_hist, label = "Train WER", color = "blue")
  plt.plot(x, valid_wer_hist, label = "Valid WER", color = "red")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.savefig('result/train/plot.png')
  plt.close()

def plot_lr_changes(lr_changes):
  x = range(0, len(lr_changes))
  plt.plot(x, lr_changes, label = "Learning Rate Changes", color = "purple")
  plt.legend()
  plt.xlabel("Iterations")
  plt.ylabel("Learning Rate")
  plt.savefig("result/train/lr_changes_plot.png")
  plt.close()

def get_scheduler(optimizer, total_iters):
  warmup_iters = int(0.1 * total_iters)
  scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
  scheduler2 = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters)
  scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
  return scheduler

def train_one_epoch(model, train_loader, loss_func, optimizer, schedular, metric, device, lr_changes, epoch=None):
  model.train()
  loss_train = AverageMeter()
  metric.reset()
  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for specs, tokens in tepoch:
      optimizer.zero_grad()
      if epoch or epoch == 0:
        tepoch.set_description(f'Epoch {epoch}')


      specs  = specs.to(device)
      tokens   = tokens.to(device)

      outputs = model(specs, tokens[:, :-1])
      loss = loss_func(outputs.permute(0, 2, 1), tokens[:, 1:])

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      if schedular:
        schedular.step()

      loss_train.update(loss.item())
      pred_sentences, input_sentences = to_sentence(outputs, tokens, model.vocab)
      metric.update(pred_sentences, input_sentences)
      lr_changes.append(optimizer.param_groups[0]['lr'])
      tepoch.set_postfix(loss=loss_train.avg, WER=metric.compute().item(), lr = optimizer.param_groups[0]['lr'])

  return model, loss_train.avg, metric.compute().item(), lr_changes

def evaluate(model, data_loader, loss_func, metric, device):
  model.eval()
  metric.reset()
  loss_valid = AverageMeter()
  for specs, tokens in data_loader:

    specs = specs.to(device)
    tokens = tokens.to(device)
    with torch.no_grad():
      outputs = model(specs, tokens[:, :-1])

    loss = loss_func(outputs.permute(0, 2, 1), tokens[:, 1:])
    loss_valid.update(loss.item())
    pred_sentences, input_sentences = to_sentence(outputs, tokens, model.vocab)

    metric.update(pred_sentences, input_sentences)

  print(f'Loss : {loss_valid.avg},          WER : {metric.compute().item()}')

  return loss_valid.avg, metric.compute().item()


def recognize_text(model, spec, vocab, device, max_length: int = 100):
  with torch.no_grad():
    input = spec.unsqueeze(0).to(device)

    start_token = vocab['<']
    tokens = torch.empty(max_length + 1, dtype=torch.long, device=device)
    tokens[0] = start_token

    itos = vocab.get_itos()

    for i in range(max_length):
      new_token = model(input, tokens[:i + 1].unsqueeze(0))[:, -1, :]
      probs = F.softmax(new_token, dim=-1)
      index = probs.argmax(dim=-1).item()

      if index == vocab['>']:
        tokens[i + 1] = index
        tokens = tokens[:i + 2]
        break

      tokens[i + 1] = index

  text = ''.join([itos[idx] for idx in tokens.tolist()])
  text = text.replace('<', '').replace('>', '')
  return text

def calculate_mer_batch(references, hypotheses):
    total_mer = 0.0
    batch_size = len(references)
    for ref, hyp in zip(references, hypotheses):
      transformation = jiwer.process_words(ref, hyp)

      S = transformation.substitutions
      D = transformation.deletions
      I = transformation.insertions
      C = len(ref.split()) - (S + D)

      mer = (S + D + I) / (S + D + I + C)
      total_mer += mer

    avg_mer = total_mer / batch_size
    return avg_mer


def calculate_ser(reference_sentences, hypothesis_sentences):
  errors = 0
  for ref, hyp in zip(reference_sentences, hypothesis_sentences):
    if ref != hyp:
      errors += 1
  return errors / len(reference_sentences)
def evaluate_for_more_info(model, data_loader, loss_func, metric, device):
  model.eval()
  metric.reset()
  cer = AverageMeter()
  mer = AverageMeter()
  ser = AverageMeter()
  bleu_score1 = torchmetrics.text.BLEUScore(n_gram=1)
  bleu_score2 = torchmetrics.text.BLEUScore(n_gram=2)
  bleu_score3 = torchmetrics.text.BLEUScore(n_gram=3)
  bleu_score4 = torchmetrics.text.BLEUScore(n_gram=4)
  loss_valid = AverageMeter()
  for specs, tokens in data_loader:

    specs = specs.to(device)
    tokens = tokens.to(device)
    with torch.no_grad():
      outputs = model(specs, tokens[:, :-1])

    loss = loss_func(outputs.permute(0, 2, 1), tokens[:, 1:])
    loss_valid.update(loss.item())
    pred_sentences, input_sentences = to_sentence(outputs, tokens, model.vocab)

    metric.update(pred_sentences, input_sentences)
    cer.update(jiwer.cer(input_sentences, pred_sentences))
    mer.update(calculate_mer_batch(input_sentences, pred_sentences))
    ser.update(calculate_ser(input_sentences, pred_sentences))
    bleu_score1.update(pred_sentences, [input_sentences])
    bleu_score2.update(pred_sentences, [input_sentences])
    bleu_score3.update(pred_sentences, [input_sentences])
    bleu_score4.update(pred_sentences, [input_sentences])

  print(f'Loss : {loss_valid.avg},          WER : {metric.compute().item()}')

  return loss_valid.avg, metric.compute().item(), cer.avg, mer.avg, ser.avg, bleu_score1.compute().item(),\
         bleu_score2.compute().item(), bleu_score3.compute().item(), bleu_score4.compute().item()