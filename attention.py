import os

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model.base_model import BaseModel
from model.vae_loss import nll_bernoulli
from utils.utils import save_cam


def attention(model: BaseModel, data_loader: DataLoader, epoch: int, save_maps: bool = True) -> None:
    model.train()
    att_scores = []
    targets = []
    recon_scores = []

    err_abs = torch.zeros(5000, 1, 20, 64, 64)

    with tqdm(total=len(data_loader), desc=f'Generating Attention Maps for epoch {epoch}') as pbar:
        for it, (data, target) in enumerate(data_loader):
            batch, ch, seq, h, w = data.shape
            data = data.to(model._device)
            reconstructions, maps = model.attention_maps(data, gradcam_pp=True)
            raw_images = (data * 255.0).squeeze().cpu().numpy()
            nll = nll_bernoulli(reconstructions.reshape(batch, seq, -1),
                                data.reshape(batch, seq, -1),
                                frame_level=True)

            err_abs[it:it + batch] = (reconstructions - data).detach()
            if target is not None:
                raw_images = mark_anomalous_frame(target, raw_images)
                targets += target.view(-1).data.tolist()
                att_scores += maps.squeeze().reshape(batch, seq, -1).mean(-1).view(-1).cpu().data.tolist()
                recon_scores += nll.view(-1).cpu().data.tolist()
            reconstructions = (reconstructions * 255.0).squeeze().cpu().detach().numpy()
            im_path = './results/'
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            base_path = im_path
            if it == 0 and save_maps:
                pbar.set_postfix_str(f'Saving in {base_path}')
                for i in range(data.shape[0]):
                    save_cam(
                        raw_images[i],
                        base_path + f"att__anom_{epoch:03d}-{i:03d}.png",
                        maps[i].squeeze().cpu().data.numpy(),
                        reconstructions[i],
                    )
            # breakpoint()
            pbar.update()
        pbar.close()

    if targets:
        auc = roc_auc_score(targets, att_scores)
        auc_mse = roc_auc_score(targets, recon_scores)
        ap = average_precision_score(targets, att_scores)
        print(f'Attention maps evaluation:\t ROC-AUC: {auc:.4f} AP: {ap:.4f},\t NLL_ROC-AUC: {auc_mse:.4f}')


def mark_anomalous_frame(target: Tensor, raw_images: Tensor,
                         mode: str = 'box') -> Tensor:
    # draw white rectangle around anomalous frame
    for i, sequence in enumerate(raw_images):
        for j, image in enumerate(sequence):
            if target[i, j].item() == 1:  # frame is anomalous
                if mode == 'dot':
                    image[-2:, -2:] = 255
                elif mode == 'box':
                    image[0, :] = 255
                    image[-1, :] = 255
                    image[:, 0] = 255
                    image[:, -1] = 255
                else:
                    raise ValueError(f'Unknown mode {mode}')

    return raw_images
