import torch
import argparse
import warnings
from utils import generate_text, get_device, blockPrint, enablePrint
from models import GPT, GPTConfig


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="GPT")

    # Input file and model weights
    parser.add_argument("--text", type=str, default=None, help="Path to the audio file for transcription.")
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

    params['vocab_size'] = len(vocab)

    # Initialize and load the GPT model
    config = GPTConfig(**params)
    model = GPT(config, vocab = vocab).to(device)
    model.load_state_dict(torch.load(args.pretrain_weight, weights_only=True))

    text = generate_text(model, args.text, vocab, device, max_length=600)
    enablePrint()  # Re-enable print statements
    print(text)  # Output the transcribed text


if __name__ == '__main__':
    args = get_arguments()
    transcript(args)