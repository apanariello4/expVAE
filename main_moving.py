
import argparse
import os
from pathlib import Path
from test import eval

import torch
from torch.optim import Adam

from model.conv3dVAE import Conv3dVAE, DebugConv3dVAE
from train import aggregate, train
from utils.utils import args, deterministic_behavior, save_checkpoint, load_moving_mnist
from torchvision.utils import save_image


def main(args: argparse.Namespace):

    train_loader, test_loader = load_moving_mnist(args)

    model = Conv3dVAE(latent_dim=args.latent_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1)

    best_test_loss = float('inf')
    resume_epoch = 0

    if args.resume:
        if args.resume == 'best':
            ckpt_dir = Path(args.ckpt_dir) / 'moving_model_best.pth'
        else:
            ckpt_dir = Path(args.ckpt_dir) / 'moving_checkpoint.pth'
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

    alpha = torch.linspace(0, 1, args.alpha_warmup)
    alpha = torch.cat((alpha, torch.ones(args.epochs - len(alpha))))

    for epoch in range(resume_epoch, args.epochs):
        train(model, train_loader, optimizer, scheduler, device, epoch, alpha[epoch])
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

        noise = torch.randn(1, args.latent_dim).to(device)
        generated = model.gen_from_noise(noise)
        generated = generated.squeeze(0).transpose(0, 1)
        save_image(generated, './generated_moving.png')


if __name__ == '__main__':
    # model = Conv3dVAE(latent_dim=32)
    # model.load_state_dict(
    #     torch.load('checkpoints/fmnist_conv3dVAE_model_best.pth')['state_dict'])
    # model.eval()
    # noise = torch.randn(1, 32)
    # generated = model.gen_from_noise(noise)
    # generated = generated.squeeze(0).transpose(0, 1)
    # save_image(generated, './generated_moving.png')

    deterministic_behavior()
    main(args())
