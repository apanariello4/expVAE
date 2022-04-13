import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.base_model import BaseModel
from model.vae_loss import nll_bernoulli
from utils.utils import save_cam


def attention(model: BaseModel, data_loader: DataLoader, epoch: int, save_maps: bool = True) -> None:
    model.train()
    att_scores = []
    targets = []
    recon_scores = []
    # err_abs = torch.zeros(5000, 1, 20, 64, 64)

    with tqdm(total=len(data_loader), desc=f'Gen. attention for ep. {epoch}') as pbar:
        for it, (data, target) in enumerate(data_loader):
            batch, _, seq, _, _ = data.shape
            data = data.to(model._device)
            reconstructions, maps = model.attention_maps(data, gradcam_pp=True)

            nll = nll_bernoulli(reconstructions.reshape(batch, seq, -1),
                                data.reshape(batch, seq, -1),
                                frame_level=True)

            data = (data * 255.0).squeeze().cpu().numpy()
            maps = maps.squeeze().detach()
            reconstructions = (reconstructions * 255.0).squeeze().detach().cpu().numpy()

            # err_abs[it:it + batch] = (reconstructions - data).detach()
            if target is not None:
                data = mark_anomalous_frame(target, data)
                targets += target.view(-1).tolist()
                # to fix when we have a (seq * seq) / 2 maps
                att_scores += maps.reshape(batch, seq, -1).mean(-1).view(-1).tolist()
                recon_scores += nll.view(-1).cpu().tolist()

            if it == 0 and save_maps:
                im_path = './results/'
                if not os.path.exists(im_path):
                    os.mkdir(im_path)
                base_path = im_path
                pbar.set_postfix_str(f'Saving in {base_path}')
                # maps have the shape (b conv ch t h w), data has the shape (b ch t h w)
                for i in range(data.shape[0]):
                    save_cam(
                        data[i],
                        base_path + f"att__anom_{epoch:03d}-{i:03d}.png",
                        maps[i].numpy(),
                        reconstructions[i],
                    )
            pbar.update()
        pbar.close()
        # np.save('./err_abs.npy', err_abs)

    if targets:
        auc = roc_auc_score(targets, att_scores)
        auc_mse = roc_auc_score(targets, recon_scores)
        ap = average_precision_score(targets, att_scores)
        print(f'Attention maps evaluation:\t ROC-AUC: {auc:.4f} AP: {ap:.4f},\t NLL_ROC-AUC: {auc_mse:.4f}')


def mark_anomalous_frame(target: Tensor, raw_images: Tensor,
                         mode: str = 'box') -> Tensor:

    if mode == 'box':
        raw_images[target == 1, 0, :] = 255
        raw_images[target == 1, -1, :] = 255
        raw_images[target == 1, :, 0] = 255
        raw_images[target == 1, :, -1] = 255
    elif mode == 'dot':
        raw_images[target == 1, -2:, -2:] = 255
    else:
        raise ValueError(f'Unknown mode {mode}')

    return raw_images
