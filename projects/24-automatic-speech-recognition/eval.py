import torch
from torch import nn
import torchmetrics
import pandas as pd
import df2img
from utils import create_dirs_test, get_device, set_seed, evaluate_for_more_info, num_trainable_params, blockPrint, enablePrint
from data import ASRDataset, get_data_loader
from models import ASRModel
import argparse
import warnings


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="FasterRCNN Network")

    # Dataset and evaluation parameters
    parser.add_argument("--data_dir", type=str, default="/content/CroppedDataset", help="Path to the dataset directory.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (test or valid).")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--n_fft", type=int, default=400, help="Number of FFT points.")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel filters.")
    parser.add_argument("--bsz", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers.")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="Path to pretrained weights.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()


def evaluate(args):
    """Evaluate the model on the specified dataset split."""
    warnings.filterwarnings("ignore")
    set_seed(args.random_seed)
    vocab = torch.load('result/vocab.pt')  # Load vocabulary
    device = get_device()

    # Prepare dataset and data loader
    dataset = ASRDataset(args.data_dir, vocab, args.split, sample_rate=args.sample_rate,
                         n_mels=args.n_mels, n_fft=args.n_fft, transform=None)
    data_loader = get_data_loader(dataset, args.bsz, False, args.num_workers)

    # Load model and hyperparameters
    with open('result/config/hyperparameters.txt', 'r') as f:
        params = eval(f.read())
    params['vocab_size'] = len(vocab)
    model = ASRModel(vocab=vocab, **params, device=device).to(device)

    # Load pretrained weights if provided
    if args.pretrain_weight:
        try:
            model.load_state_dict(torch.load(args.pretrain_weight, weights_only=True, map_location=device))
        except:
            model = torch.load(args.pretrain_weight, map_location=device).to(device)

    # Print model details
    print('Vocab size:', len(vocab))
    print(f'{num_trainable_params(model)} million parameters')
    print(f'Parameters: \n{params}')

    # Define loss function and metrics
    loss_func = nn.CrossEntropyLoss(ignore_index=vocab['='])
    metric = torchmetrics.text.WordErrorRate().to(device)

    # Evaluate the model
    loss, wer, cer, mer, ser, bleu1, bleu2, bleu3, bleu4 = evaluate_for_more_info(model, data_loader, loss_func, metric, device)

    # Prepare results for saving
    indexes = ['Parameters', 'Loss', 'WER (Word Error Rate)', 'CER (Character Error Rate)',
               'MER (Match Error Rate)', 'SER (Sentence Error Rate)',
               'BLEU (n_gram=1)', 'BLEU (n_gram=2)', 'BLEU (n_gram=3)', 'BLEU (n_gram=4)']
    results = [num_trainable_params(model), round(loss, 5), round(wer, 3), round(cer, 3),
               round(mer, 3), round(ser, 3), round(bleu1, 3), round(bleu2, 3),
               round(bleu3, 3), round(bleu4, 3)]

    # Save results as image and CSV
    blockPrint()
    info = pd.DataFrame(results, index=indexes, columns=['ASRModel'])
    fig = df2img.plot_dataframe(info, fig_size=(600, 250))
    df2img.save_dataframe(fig=fig, filename=f"result/{args.split}/info.png")
    info.to_csv(f"result/{args.split}/info.csv")
    enablePrint()


if __name__ == '__main__':
    args = get_arguments()
    create_dirs_test(args.split)  # Create directories for saving results
    evaluate(args)