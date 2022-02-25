import argparse
from ast import parse
import os
from pathlib import Path

import torch
from PIL import Image

from model.convVAE import ConvVAE
from model.gradCAM import gradCAM
from utils.utils import deterministic_behavior, load_mnist_one_class, save_cam, load_fmnist_one_class
import numpy as np
import cv2
from torch import Tensor


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset-path', type=str, default='./data/mnist')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--outlier-class', type=int, default=6)
    parser.add_argument('--distr-difference', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='fmnist')
    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvVAE(
        args.latent_dim).to(device)
    if args.dataset == 'fmnist':
        _, test_loader = load_fmnist_one_class(args.outlier_class, args)
    elif args.dataset == 'mnist':
        _, test_loader = load_mnist_one_class(args.outlier_class, args)

    model_path = Path(args.ckpt_dir) / f'{args.dataset}_checkpoint.pth'
    checkpoint = torch.load(model_path)
    mu_avg = checkpoint['mu_avg'].to(device)
    var_avg = checkpoint['var_avg'].to(device)
    model.load_state_dict(checkpoint['state_dict'])
    print(
        f"Model loaded\n Inlier class: {checkpoint['inlier_class']}, Outlier class: {args.outlier_class}\n",
        f"Train Dataset: {checkpoint['dataset']}, Test Dataset: {args.dataset}")
    att = gradCAM(model, device)

    test_index = 0
    for x, _ in test_loader:

        batch_size = x.size(0)
        x = x.to(device)
        x_hat, mu, logvar = att.forward(x)

        if args.distr_difference:
            mu = mu_avg - mu
            logvar = torch.log(var_avg + torch.exp(logvar))

        att_map, att_map_loss = att.get_attention_map(x, x_hat, mu, logvar)

        for img in range(batch_size):
            raw_image = (x[img] * 255.0).squeeze().cpu().numpy()
            im_path = './results/'
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            base_path = im_path + f"{test_index}-{args.outlier_class}-"

            recon_error = torch.mean(torch.abs(x[img] - x_hat[img]))

            save_cam(
                raw_image,
                base_path + "att.png",
                att_map[img].squeeze().cpu().data.numpy(),
                att_map_loss[img].squeeze().cpu().data.numpy())
            test_index += 1


if __name__ == '__main__':
    deterministic_behavior()
    main(args())
