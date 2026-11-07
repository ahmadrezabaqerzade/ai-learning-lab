import torch
import torch.nn.functional as F
import torchmetrics.text
from torch.optim.lr_scheduler import  CosineAnnealingLR, LinearLR, SequentialLR
import jiwer
from rouge_score import rouge_scorer
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random
import sys
import os
import time

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
    model_params = dict(block_size = 1024, vocab_size = 50304 , n_layer = 6, n_head = 6, n_embed = 384, dropout = 0.2, bias = False)
    with open('result/config/hyperparameters.txt', 'w') as f:
      f.write(str(model_params))
def create_dirs_test(split):

  if not os.path.exists('result/'):
    os.makedirs('result/')

  if not os.path.exists(f'result/{split}'):
    os.makedirs(f'result/{split}')
def create_dirs_visualize():

  if not os.path.exists('result/'):
    os.makedirs('result/')

  if not os.path.exists(f'result/visualize'):
    os.makedirs(f'result/visualize')

def print_details(model_num_params, model_config, num_sequence_train, num_sequence_valid):
  print("Model:")
  print("num trainable parameters:       ", model_num_params, ' Million')
  print(f"block_size = {model_config['block_size']}      n_layer = {model_config['n_layer']}           vocab_size = {model_config['vocab_size']}\n"
        f"n_head = {model_config['n_head']}         n_embed = {model_config['n_embed']}                 dropout = {model_config['dropout']}\n"
        f"bias = {model_config['bias']}")
  print('------------------------------------------------------------------------------')
  print("Dataset:")
  print(f"{num_sequence_train} sequences for train")
  print(f"{num_sequence_valid} sequences for valid")
  print(f"vocab size = {model_config['vocab_size']}")
  print('------------------------------------------------------------------------------')

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
    best_per, best_epoch, \
    train_loss_hist, \
    train_per_hist, \
    valid_loss_hist, \
    valid_per_hist, w_op_sch = torch.load("result/train/info.pt", weights_only=True, map_location=device)
    start, stop = len(train_loss_hist), len(train_loss_hist) + num_epochs
    lr = lr_changes[-1]
  else:
    train_loss_hist = []
    train_per_hist = []
    valid_loss_hist = []
    valid_per_hist = []
    best_per = torch.inf
    best_epoch = 0
    wait = 0
    start, stop = 0, num_epochs
    lr = learning_rate
    lr_changes = []
    w_op_sch = None

  return train_loss_hist, train_per_hist, valid_loss_hist, valid_per_hist, best_per, best_epoch, wait, start, stop, lr, lr_changes, w_op_sch


def to_sentence(preds, inputs, vocab):
  vocab_list = vocab.get_itos()
  def indexes_to_sentence(indexes):
    sentence = ''.join([vocab_list[index] for index in indexes])
    return sentence

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
  train_per_hist, \
  valid_loss_hist, \
  valid_per_hist, _ = info
  plt.figure(figsize = (12, 8))
  x = range(0, len(train_per_hist))
  plt.subplot(1, 2, 1)
  plt.plot(x, train_loss_hist, label = "Train Loss", color = "blue")
  plt.plot(x, valid_loss_hist, label = "Valid Loss", color = "red")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.subplot(1, 2, 2)
  plt.plot(x, train_per_hist, label = "Train Perplexity", color = "blue")
  plt.plot(x, valid_per_hist, label = "Valid Perplexity", color = "red")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Perplexity")
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
    for input, target in tepoch:
      optimizer.zero_grad()
      if epoch or epoch == 0:
        tepoch.set_description(f'Epoch {epoch}')


      input  = input.to(device)
      target   = target.to(device)

      output = model(input)
      loss = loss_func(output.view(-1, output.shape[-1]), target.view(-1))

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
      optimizer.step()
      if schedular:
        schedular.step()

      loss_train.update(loss.item())
      metric.update(output, target)
      lr_changes.append(optimizer.param_groups[0]['lr'])
      tepoch.set_postfix(loss=loss_train.avg, Perplexity=metric.compute().item(), lr = optimizer.param_groups[0]['lr'])

  return model, loss_train.avg, metric.compute().item(), lr_changes

def evaluate(model, data_loader, loss_func, metric, device):
  model.eval()
  metric.reset()
  loss_valid = AverageMeter()
  for input, target in data_loader:

    input  = input.to(device)
    target = target.to(device)
    with torch.no_grad():
      output = model(input)

    loss = loss_func(output.view(-1, output.shape[-1]), target.view(-1))
    loss_valid.update(loss.item())

    metric.update(output, target)

  print(f'Loss : {loss_valid.avg},          Perplexity : {metric.compute().item()}')

  return loss_valid.avg, metric.compute().item()


def generate_text(model, text, vocab, device, temp=1.0, max_length: int = 10):
  with torch.no_grad():

    tokens = torch.empty(max_length + 1, dtype=torch.long, device=device)
    indices = [vocab[ch] for ch in text]
    tokens[0:len(indices)] = torch.LongTensor(indices)

    itos = vocab.get_itos()

    for i in range(len(indices)-1, max_length):
      new_token = model(tokens[:i + 1].unsqueeze(0))[:, -1, :]/temp
      probs = F.softmax(new_token, dim=-1)
      index =  torch.multinomial(probs, num_samples=1)
      tokens[i + 1] = index

  text = ''.join([itos[idx] for idx in tokens.tolist()])
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
  s = time.time()
  model.eval()
  metric.reset()
  cer = AverageMeter()
  mer = AverageMeter()
  ser = AverageMeter()
  rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  rouge_p_1 = AverageMeter()
  rouge_p_2 = AverageMeter()
  rouge_p_L = AverageMeter()
  rouge_r_1 = AverageMeter()
  rouge_r_2 = AverageMeter()
  rouge_r_L = AverageMeter()
  rouge_f_1 = AverageMeter()
  rouge_f_2 = AverageMeter()
  rouge_f_L = AverageMeter()
  bleu_score1 = torchmetrics.text.BLEUScore(n_gram=1)
  bleu_score2 = torchmetrics.text.BLEUScore(n_gram=2)
  bleu_score3 = torchmetrics.text.BLEUScore(n_gram=3)
  bleu_score4 = torchmetrics.text.BLEUScore(n_gram=4)
  loss_valid = AverageMeter()
  for input, target in data_loader:
    input = input.to(device)
    target = target.to(device)
    with torch.no_grad():
      output = model(input)

    loss = loss_func(output.view(-1, output.shape[-1]), target.view(-1))
    loss_valid.update(loss.item())
    pred_sentences, input_sentences = to_sentence(output, target, model.vocab)

    metric.update(output, target)
    cer.update(jiwer.cer(input_sentences, pred_sentences))
    mer.update(calculate_mer_batch(input_sentences, pred_sentences))
    ser.update(calculate_ser(input_sentences, pred_sentences))
    bleu_score1.update(pred_sentences, [input_sentences])
    bleu_score2.update(pred_sentences, [input_sentences])
    bleu_score3.update(pred_sentences, [input_sentences])
    bleu_score4.update(pred_sentences, [input_sentences])
    for input_s, pred_s in zip(input_sentences, pred_sentences):
      sc = rouge.score(input_s, pred_s)
      rouge_p_1.update(sc['rouge1'].precision)
      rouge_p_2.update(sc['rouge2'].precision)
      rouge_p_L.update(sc['rougeL'].precision)
      rouge_r_1.update(sc['rouge1'].recall)
      rouge_r_2.update(sc['rouge2'].recall)
      rouge_r_L.update(sc['rougeL'].recall)
      rouge_f_1.update(sc['rouge1'].fmeasure)
      rouge_f_2.update(sc['rouge2'].fmeasure)
      rouge_f_L.update(sc['rougeL'].fmeasure)

  print(f'Loss : {loss_valid.avg},          Perplexity : {metric.compute().item()}     Evaluation Time :  {time.time() - s}')

  return loss_valid.avg, metric.compute().item(), cer.avg, mer.avg, ser.avg, bleu_score1.compute().item(),\
         bleu_score2.compute().item(), bleu_score3.compute().item(), bleu_score4.compute().item(), rouge_p_1.avg, \
         rouge_p_2.avg, rouge_p_L.avg, rouge_r_1.avg, rouge_r_2.avg, rouge_r_L.avg, rouge_f_1.avg, rouge_f_2.avg, rouge_f_L.avg