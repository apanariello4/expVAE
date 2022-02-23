
import argparse
import os
from pathlib import Path
from test import eval

import torch
from torch.optim import Adam

from model.convVAE import ConvVAE
from train import aggregate, train
from utils.utils import args, deterministic_behavior, save_checkpoint, load_mnist_one_class, load_fmnist_one_class
from torchvision.utils import save_image


def main(args: argparse.Namespace):

    if args.dataset == 'mnist':
        train_loader, test_loader = load_mnist_one_class(
            class_id=args.one_class, args=args)
    elif args.dataset == 'fmnist':
        train_loader, test_loader = load_fmnist_one_class(
            class_id=args.one_class, args=args)

    model = ConvVAE(latent_dim=args.latent_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Dataset: {args.dataset}, Inlier class: {args.one_class}")
    best_test_loss = float('inf')
    resume_epoch = 0

    if args.resume:
        if args.resume == 'best':
            ckpt_dir = Path(args.ckpt_dir) / f'{args.dataset}_model_best.pth'
        else:
            ckpt_dir = Path(args.ckpt_dir) / f'{args.dataset}_checkpoint.pth'
        assert os.path.exists(ckpt_dir)
        print("Loading checkpoint")
        checkpoint = torch.load(
            os.path.join(
                args.ckpt_dir,
                'checkpoint.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch']
        print("Loaded checkpoint")

    for epoch in range(resume_epoch, args.epochs):
        train(model, train_loader, optimizer, device, epoch)
        mu_avg, var_avg = aggregate(model, train_loader, device)
        test_loss = eval(model, device, test_loader)
        print(f'Epoch {epoch + 1} val_loss: {test_loss}')

        if test_loss < best_test_loss or epoch == args.epochs - 1:
            best_test_loss = test_loss
            save_checkpoint(
                model,
                epoch,
                optimizer, (mu_avg, var_avg),
                is_last=epoch == args.epochs - 1,
                outdir=args.ckpt_dir, args=args)

    noise = torch.randn(args.batch_size, args.latent_dim).to(device)
    generated = model.gen_from_noise(noise)
    save_image(generated, './generated.png')


if __name__ == '__main__':
    deterministic_behavior()
    main(args())
