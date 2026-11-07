import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
import argparse
import warnings
from utils import recognize_text, get_device, blockPrint, enablePrint
from models import ASRModel


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Speech to Text")

    # Input file and model weights
    parser.add_argument("--file_path", type=str, default=None, help="Path to the audio file for transcription.")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="Path to pretrained model weights.")

    return parser.parse_args()


def transcript(args):
    """Transcribe an audio file using a pretrained ASR model."""
    warnings.filterwarnings("ignore")
    blockPrint()  # Suppress unnecessary print statements

    # Load device and vocabulary
    device = get_device()
    vocab = torch.load('result/vocab.pt')

    # Load model hyperparameters
    with open('result/config/hyperparameters.txt', 'r') as f:
        params = eval(f.read())
    params['vocab'] = vocab
    params['vocab_size'] = len(vocab)

    # Initialize and load the ASR model
    model = ASRModel(**params, device=device).to(device)
    model.load_state_dict(torch.load(args.pretrain_weight, weights_only=True))

    # Load and preprocess the audio file
    waveform, sample_rate = torchaudio.load(args.file_path)
    spec = nn.Sequential(
        T.Resample(orig_freq=sample_rate, new_freq=16000),
        T.MelSpectrogram(n_mels=80, n_fft=400)
    ).requires_grad_(False)(waveform)
    spec = torch.log10(spec + 1e-9)  # Log scaling
    spec = (spec - spec.mean()) / spec.std()  # Normalize

    # Recognize text from the spectrogram
    text = recognize_text(model, spec, vocab, device)
    enablePrint()  # Re-enable print statements
    print(text)  # Output the transcribed text


if __name__ == '__main__':
    args = get_arguments()
    transcript(args)