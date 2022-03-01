
import argparse
import os
import time
from pathlib import Path
from test import eval, eval_anom

import torch
from torch.optim import Adam
from torchvision.utils import save_image

import wandb
from model.conv3dVAE import Conv3dVAE
from train import aggregate, train
from utils.utils import (args, deterministic_behavior, load_moving_mnist,
                         save_checkpoint)


def main(args: argparse.Namespace):

    train_loader, test_loader, anom_loader = load_moving_mnist(args)

    model = Conv3dVAE(latent_dim=args.latent_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epochs // args.lr_steps, gamma=0.1)

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

    if args.log:
        name = args.name + '_' if args.name else ''
        wandb.init(project='exp_vae', name=f"{name}ld_{args.latent_dim}_lr_{args.lr}", config=args)
        wandb.watch(model)

    for epoch in range(resume_epoch, args.epochs):
        t0 = time.time()
        train(model, train_loader, optimizer, scheduler, device, epoch, alpha[epoch])
        mu_avg, var_avg, min_max_loss = aggregate(model, train_loader, device)
        test_loss = eval(model, device, test_loader, epoch)
        test_loss = .0
        roc_auc, ap = eval_anom(model, device, anom_loader, epoch, min_max_loss)
        print(f'Epoch {epoch} val_loss: {test_loss:.4f}\tROC-AUC: {roc_auc:.4f} AP: {ap:.4f}\tEpoch time {time.time() - t0:.4f}')

        if args.save_checkpoint:
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

        if wandb.run:
            wandb.log({
                'mu_avg': mu_avg, 'var_avg': var_avg,
                'img_from_noise': wandb.Image(generated, caption='Generated from noise')
            }, step=epoch)


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
