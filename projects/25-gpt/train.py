import torch
import torchmetrics
from torch import nn, optim
from utils import create_dirs, get_device, create_files, set_seed, \
    train_one_epoch, evaluate, num_trainable_params, plot_and_save_result, plot_lr_changes, get_scheduler, get_changes, print_details
from data import TextGenerationDataset, get_data_loader, get_vocab
from models import GPT, GPTConfig
import argparse
import warnings


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="GPT")

    # Dataset and training parameters
    parser.add_argument("--data_dir", type=str, default="/content/Dataset", help="Path to the dataset directory.")
    parser.add_argument("--bsz", type=int, default=16, help="Batch size.")
    parser.add_argument("--create_vocab", type=str, default="False", help="Create or recreate vocabulary.")
    parser.add_argument("--seq_length", type=int, default=80, help="length of sequences to be train.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers.")

    # Optimizer and learning parameters
    parser.add_argument("--learning_rate", type=float, default=0.2, help="Base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--nesterov", type=str, default='True', help="Use Nesterov momentum.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--clip", type=float, default=0.5, help="Gradient clipping value.")

    # Training control
    parser.add_argument("--num_epochs", type=int, default=1000, help="Total training epochs.")
    parser.add_argument("--early_stop", type=int, default=1000, help="Early stopping patience.")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="Path to pretrained weights.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--continue_train", type=str, default="False", help="Continue training from checkpoint.")

    return parser.parse_args()


def train(args):
    """Main training function."""
    warnings.filterwarnings("ignore")
    set_seed(args.random_seed)

    # Data transforms and dataset preparation
    if eval(args.create_vocab) == True:
        print('create vocab')
        vocab = get_vocab(args.data_dir + '/train.txt')
        torch.save(vocab, 'result/vocab.pt')
        print('done')
    else:
        print('load vocab')
        vocab = torch.load('result/vocab.pt')
        print('done')

    train_dataset = TextGenerationDataset(args.data_dir, vocab, 'train', args.seq_length)
    valid_dataset = TextGenerationDataset(args.data_dir, vocab, 'valid', args.seq_length)

    train_loader = get_data_loader(train_dataset, args.bsz, True, args.num_workers)
    valid_loader = get_data_loader(valid_dataset, args.bsz, False, args.num_workers)

    # Model setup
    device = get_device()
    with open('result/config/hyperparameters.txt', 'r') as f:
        params = eval(f.read())

    params['vocab_size'] = len(vocab)
    config = GPTConfig(**params)
    model = GPT(config, vocab = vocab).to(device)

    if args.pretrain_weight:
        try:
            model.load_state_dict(torch.load(args.pretrain_weight, weights_only=True, map_location=device))
        except:
            model = torch.load(args.pretrain_weight, map_location=device).to(device)

    print_details(num_trainable_params(model), params, len(train_dataset), len(valid_dataset))

    # Training setup
    train_loss_hist, train_per_hist, valid_loss_hist, valid_per_hist, best_per, best_epoch, wait, start, stop, lr, lr_changes, w_op_sch = \
        get_changes(args.continue_train, model, device, args.num_epochs, args.learning_rate)
    if not lr_changes:
        lr_changes = [args.learning_rate]

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=eval(args.nesterov))
    scheduler = get_scheduler(optimizer, args.num_epochs * len(train_loader))

    if train_loss_hist:
        optimizer.load_state_dict(w_op_sch['opt params'])
        scheduler.load_state_dict(w_op_sch['sch params'])

    loss_func = nn.CrossEntropyLoss()
    metric = torchmetrics.text.Perplexity().to(device)
    # Training loop
    for epoch in range(start, stop):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, train_loss, train_per, lr_changes = train_one_epoch(model, train_loader, loss_func,
                                                                   optimizer, scheduler, metric, device, lr_changes,
                                                                   epoch)
        valid_loss, valid_per = evaluate(model, valid_loader, loss_func, metric, device)

        # Logging and saving
        train_loss_hist.append(train_loss)
        train_per_hist.append(train_per)
        valid_loss_hist.append(valid_loss)
        valid_per_hist.append(valid_per)

        if valid_per < best_per:
            torch.save(model.state_dict(), 'result/model/best_model.pt')
            print('Model saved!')
            wait = 0
            best_per = valid_per
            best_epoch = epoch

        torch.save(model.state_dict(), 'result/model/last_model.pt')
        torch.save([lr_changes, wait, best_per, best_epoch, train_loss_hist, train_per_hist,
                    valid_loss_hist, valid_per_hist, {"sch params": scheduler.state_dict(),
                                                      "opt params": optimizer.state_dict()}], "result/train/info.pt")

        plot_and_save_result(torch.load("result/train/info.pt", weights_only=True, map_location=device))
        plot_lr_changes(lr_changes)

        if epoch == stop - 1 or wait == args.early_stop:
            print(f'Best epoch: {best_epoch}')
            if wait == args.early_stop:
                break


if __name__ == '__main__':
    create_dirs()
    create_files()
    __spec__ = None
    args = get_arguments()
    train(args)