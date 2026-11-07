import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

def get_vocab(text_file_address):
    with open(text_file_address, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = build_vocab_from_iterator([sorted(list(set(text)))], min_freq=0,
                                     specials=[' '])
    vocab.set_default_index(0)  # Set default index for out-of-vocabulary tokens
    return vocab


class TextGenerationDataset(Dataset):
    def __init__(self, data_dir, vocab, phase, block_size=50):
        super(TextGenerationDataset, self).__init__()
        self.data_dir = data_dir
        self.block_size = block_size + 1
        with open(data_dir +  '/' + phase + '.txt', 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.vocab = vocab

    def __getitem__(self, index):
        # Convert text to token indices
        text = self.data[index*self.block_size : index*self.block_size + self.block_size]
        text_indexes = torch.LongTensor([self.vocab[tok] for tok in text])
        input  = text_indexes[:-1]
        target = text_indexes[1:]
        return input, target

    def __len__(self):
        return int((len(self.data)-self.block_size)/self.block_size)

def get_data_loader(dataset, batch_size, shuffle, num_workers):
    """Create and return a DataLoader for the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
