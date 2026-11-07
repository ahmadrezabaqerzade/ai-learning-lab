import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from augmentation import LogarithmicScaleSpec, NormalizeSpec

def get_vocab(data_dir):
    """Build and return a vocabulary from the training text data."""
    train_sen = pd.read_csv(data_dir + '/split/train-ann.csv')['text']
    vocab = build_vocab_from_iterator(train_sen.apply(lambda x: x.lower()), min_freq=0,
                                     specials=['=', '#', '<', '>'])
    vocab.set_default_index(1)  # Set default index for out-of-vocabulary tokens
    return vocab


def read_split_files(data_dir):
    """Read and return train, validation, and test data splits."""
    train_df = pd.read_csv(data_dir + '/split/train-ann.csv')
    valid_df = pd.read_csv(data_dir + '/split/valid-ann.csv')
    test_df = pd.read_csv(data_dir + '/split/test-ann.csv')
    return train_df, valid_df, test_df


class ASRDataset(Dataset):
    """Dataset class for Automatic Speech Recognition (ASR)."""
    def __init__(self, data_dir, vocab, phase, sample_rate=16000, n_mels=80, n_fft=512, transform=None):
        super(ASRDataset, self).__init__()
        self.data_dir = data_dir
        self.data = read_split_files(self.data_dir)[0 if phase == 'train' else (1 if phase == 'valid' else 2)]
        self.transform = transform
        self.vocab = vocab
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        """Load and preprocess an audio sample and its corresponding text."""
        audio = self.data['audio'].iloc[index]
        audio = audio if '.wav' in audio else audio + '.wav'
        waveform, sample_rate = torchaudio.load(self.data_dir + '/audios/' + audio)
        # Compute Mel spectrogram
        spec = nn.Sequential(
            T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate),
            T.MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft)
        ).requires_grad_(False)(waveform)
        if self.transform:
            norm_log_spec = self.transform(spec).detach()
        else:
            norm_log_spec = nn.Sequential(LogarithmicScaleSpec(), NormalizeSpec())(spec).detach()

        # Convert text to token indices
        text = '< ' + self.data['text'].iloc[index] + ' >'
        text_indexes = torch.LongTensor([self.vocab[ch.lower()] for ch in text])

        return norm_log_spec, text_indexes

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data.shape[0]


def collate_fn(batch):
    """Collate function to pad spectrograms and text sequences to the same length."""
    max_len = max(sam[0].shape[2] for sam in batch)
    specs = torch.cat([F.pad(sam[0], (0, max_len - sam[0].shape[2])).unsqueeze(0) for sam in batch], dim=0)
    texts = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
    return specs, texts


def get_data_loader(dataset, batch_size, shuffle, num_workers):
    """Create and return a DataLoader for the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)