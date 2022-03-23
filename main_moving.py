
import argparse
import datetime
import os
import time
from test import eval, eval_anom

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torchvision.utils import save_image

import wandb
from model.Bi_conVRNN import BidirectionalConVRNN
from model.conv3dVAE import Conv3dVAE
from model.conVRNN import ConVRNN
from model.DSVAE import DisentangledVAE
from model.gradCAM import GradCAM
from model.LoCOVAE import LoCOVAE
from train import aggregate, train
from utils.dataset_loaders import load_moving_mnist
from utils.utils import args, deterministic_behavior, get_alpha_scheduler, save_checkpoint, save_cam


def main(args: argparse.Namespace):

    train_loader, test_loader, anom_loader = load_moving_mnist(args)

    architecture = {'loco': LoCOVAE,
                    'conv3d': Conv3dVAE,
                    'vrnn': ConVRNN,
                    'bivrnn': BidirectionalConVRNN,
                    'dsvae': DisentangledVAE}[args.model]

    if args.model in ['loco', 'conv3d']:
        model = architecture(latent_dim=args.latent_dim, activation=args.activation)
        if args.recon_func == 'bce':
            model.decoder.add_module("sigmoid", nn.Sigmoid())

    elif args.model in ['vrnn', 'bivrnn']:
        model = architecture(h_dim=512, latent_dim=args.latent_dim, activation=args.activation)

    elif args.model == 'dsvae':
        model = architecture()

    print(f'Model: {model.name}, num params: {model.count_parameters:,}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.epochs // args.lr_steps)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                T_0=(args.epochs + 1) // 2, T_mult=1)

    best_test_loss = float('inf')
    resume_epoch = 0

    if args.resume:
        assert os.path.exists(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint {args.resume.split('/')[-1]}")

    now = datetime.datetime.now().strftime("%m%d%H%M")
    if args.log:
        name = args.name + '_' if args.name else ''
        wandb.init(project='exp_vae', name=f"{name}{model.name}_{args.activation}_{now}", config=args)
        wandb.watch(model)

    recon_func = args.recon_func
    if args.attention:
        gradcam = GradCAM(model)

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
        print(f'Epoch {epoch} val_loss: {test_loss:.4g} \tROC-AUC: {roc_auc:.4g} AP: {ap:.4g}\tEpoch time {(time.time() - t0)/60:.4g} m')

    if args.attention:
        # model.eval()
        sequence = next(iter(test_loader))[0].to(device)
        x_hat, mu, logvar = gradcam.forward(sequence)

        maps = gradcam.get_attention_map(sequence, x_hat, mu, logvar, target_layer='phi_x.2.downsample')
        raw_image = (sequence * 255.0).squeeze().cpu().numpy()
        im_path = './results/'
        if not os.path.exists(im_path):
            os.mkdir(im_path)
        base_path = im_path + args.name

        save_cam(
            raw_image,
            base_path + f"att{epoch}.png",
            maps.squeeze().cpu().data.numpy(),
            # loss_maps.squeeze().cpu().data.numpy()
        )

        if args.save_checkpoint and not args.test_only:
            if test_loss < best_test_loss or epoch == args.epochs - 1:
                best_test_loss = test_loss
                save_checkpoint(
                    model,
                    epoch,
                    optimizer,  # (mu_avg, var_avg),
                    is_last=epoch == args.epochs - 1,
                    args=args, time=now)

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
