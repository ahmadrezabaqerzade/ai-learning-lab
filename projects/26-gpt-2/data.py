import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset as hf_dataset
import numpy as np

class TinyStoriesDataset(IterableDataset):
  def __init__(self, data_path, seq_length, group_size):
    self.data = hf_dataset.from_parquet(f"{data_path}/*.parquet")
    self.seq_length = seq_length
    self.group_size = group_size
  def __iter__(self):
    buffer = []
    for i in range(0, len(self.data), self.group_size):
      buffer += list(eval(','.join(self.data[i:i+self.group_size]['ids'])))
      yield from (
                  torch.from_numpy(np.array(buffer[j:j+self.seq_length+1], dtype = np.int64))
                  for j in range(0, len(buffer) - self.seq_length, self.seq_length)
      )

      buffer = buffer[(len(buffer)//self.seq_length)*self.seq_length:]

def get_data_loader(dataset, bs, nw):
  d_loader = DataLoader(dataset, batch_size = bs, num_workers = nw)
  return d_loader