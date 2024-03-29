
import argparse
import datetime
import os
import time
from main_mil import main_mil
from test import eval, eval_anom

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

import wandb
from attention import attention
from model.Bi_conVRNN import BidirectionalConVRNN
from model.conv3dVAE import Conv3dVAE
from model.conVRNN import ConVRNN
from model.DSVAE import DisentangledVAE
from model.LoCOVAE import LoCOVAE
from train import aggregate, train
from utils.dataset_loaders import load_moving_mnist
from utils.utils import args, deterministic_behavior, get_alpha_scheduler, get_scheduler, save_checkpoint
import torchvision
from model.r2p1d import SmallResNet, get_model, generate_model
from main_resnet import main_resnet
from model.semisup_conVRNN import SemiSupConVRNN


def main(args: argparse.Namespace):

    architecture = {'loco': LoCOVAE,
                    'conv3d': Conv3dVAE,
                    'vrnn': ConVRNN,
                    'bivrnn': BidirectionalConVRNN,
                    'dsvae': DisentangledVAE,
                    'resnet': generate_model,
                    'mil': SemiSupConVRNN}[args.model]

    print('Loading model...')
    if args.model in ['loco', 'conv3d']:
        model = architecture(latent_dim=args.latent_dim, activation=args.activation)
        if args.recon_func == 'bce':
            model.decoder.add_module("sigmoid", nn.Sigmoid())

    elif args.model in ['vrnn', 'bivrnn', 'mil']:
        model = architecture(h_dim=512, latent_dim=args.latent_dim, activation=args.activation)
        if args.model == 'mil':
            main_mil(model, args)

    elif args.model == 'dsvae':
        model = architecture()

    elif args.model == 'resnet':
        model = SmallResNet()  # architecture(model_depth=10, n_input_channels=3, n_classes=2)
        # model = torchvision.models.video.r2plus1d_18(pretrained=True, num_classes=400, progress=True)
        # model.fc = nn.Linear(512, 2)
        # model.name = 'R2P1D-18'
        num_params = sum(p.numel() for p in model.parameters())
        num_params_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Params: Total {num_params:,}, learnable {num_params_learn:,}')
        main_resnet(model, args)

    print(f'\nModel: {model.name}, num params: {model.count_parameters:,}, activation: {args.activation}\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, test_loader, anom_loader = load_moving_mnist(args)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(optimizer, args)
    best_test_loss = float('inf')
    resume_epoch = 0

    if args.resume:
        assert os.path.exists(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint {args.resume.split('/')[-1]} (epoch {resume_epoch})\n")

    now = datetime.datetime.now().strftime("%m%d%H%M")
    if args.log:
        name = args.name + '_' if args.name else ''
        wandb.init(project='exp_vae', name=f"{name}{model.name}_{args.activation}_{now}", config=args)
        wandb.watch(model)

    recon_func = args.recon_func

    alpha = get_alpha_scheduler(args)

    for epoch in range(resume_epoch, args.epochs):
        t0 = time.time()
        min_max_loss = None
        if not args.test_only:
            train(model, train_loader, optimizer, scheduler, device, epoch, recon_func, alpha[epoch])

            if args.model not in ['bivrnn', 'dsvae']:
                _, _, min_max_loss = aggregate(model, train_loader, device, recon_func)

        test_loss = eval(model, device, test_loader, epoch, recon_func)
        roc_auc, ap = eval_anom(model, device, anom_loader, epoch, min_max_loss, recon_func)
        print(f'Epoch {epoch} val_loss: {test_loss:.4g} \tROC-AUC: {roc_auc:.4g} AP: {ap:.4g}\tEpoch time {(time.time() - t0)/60:.4g}m')

        if args.save_checkpoint and not args.test_only:
            if test_loss < best_test_loss or epoch == args.epochs - 1:
                best_test_loss = test_loss
                save_checkpoint(
                    model,
                    epoch,
                    optimizer,  # (mu_avg, var_avg),
                    is_last=epoch == args.epochs - 1,
                    args=args, time=now)

        if args.attention:
            attention(model, anom_loader, epoch)

        with torch.no_grad():
            model.train()
            generated = model.sample()
        # must be 20,1,64,64
        save_image(generated, './generated_moving.png')

        if wandb.run:
            wandb.log({
                # 'mu_avg': mu_avg, 'var_avg': var_avg,
                'img_from_noise': wandb.Image(generated, caption=f'Generated from noise ep {epoch}'),
            }, step=epoch)

        if args.test_only:
            break


if __name__ == '__main__':
    # model = ConVRNN(512, 128, 'elu')
    # model.load_state_dict(
    #     torch.load('/home/nello/expVAE/checkpoints/shallow__conVRNN_03081041best.pth')['state_dict'])
    # model.eval()
    # generated = model.sample()
    # save_image(generated, './generated_eval.png')
    # model.train()
    # generated = model.sample()
    # save_image(generated, './generated_train.png')

    deterministic_behavior()
    main(args())
