import argparse
from typing import Tuple
import torch
import wandb
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.MovingMNIST import MovingMNIST
from model.conVRNN import ConVRNN

from model.vae_loss import vae_loss, vae_loss_normalized
from sklearn.metrics import roc_auc_score, average_precision_score


def save_one_recon_batch(model, device, test_loader, epoch):
    with torch.no_grad():
        for x, _ in test_loader:
            model.eval()
            x = x.to(device)

            x_hat, _, _ = model(x)

            imgs = gen_one_recon_img(x[0], x_hat[0])

            save_image(imgs, f'ep-{epoch}_recon_moving.png', nrow=10)

            break


def gen_one_recon_img(x: Tensor, x_hat: Tensor) -> Tensor:
    assert x.max() <= 1.0 and x.min() >= 0.0, "x has not been normalized"
    assert x_hat.max() <= 1.0 and x_hat.min() >= 0.0, "x_hat has not been normalized"

    imgs = torch.cat([x.transpose(0, 1), x_hat.transpose(0, 1)], dim=0)
    return imgs


def eval(model, device: torch.device, test_loader: DataLoader, epoch: int, recon_func: str) -> float:
    model.eval()
    test_loss = 0.0
    total_recon_err, total_kld_err = 0.0, 0.0
    num_samples = 0
    loss_function = model.get_loss_function(recon_func=recon_func)

    with torch.no_grad(), tqdm(total=len(test_loader), desc='Eval ') as pbar:
        for data, _ in test_loader:

            data = data.to(device)

            batch_size = data.shape[0]
            num_samples += batch_size

            x_recon, *distribution = model(data)

            loss, recon_err, kld_err = loss_function(x_recon, data, *distribution)

            test_loss += loss.item()
            total_recon_err += recon_err.item()
            total_kld_err += kld_err.item()
            pbar.set_postfix(loss=test_loss / num_samples, recon_err=total_recon_err / num_samples, kld_err=total_kld_err / num_samples)
            pbar.update()
        pbar.close()
        if isinstance(x_recon, list):
            x_recon = x_recon[0]
        imgs = gen_one_recon_img(data[0], x_recon[0])

        if wandb.run:
            w_imgs = wandb.Image(imgs, caption=f'Reconstruction ep {epoch}')
            wandb.log({'test_loss': test_loss / num_samples,
                       'test_recon_err': total_recon_err / num_samples,
                       'test_kld_err': total_kld_err / num_samples,
                       'reconstruction': w_imgs}, step=epoch)

    test_loss /= len(test_loader.dataset)
    return test_loss


def eval_anom(model, device: torch.device, anom_loader: DataLoader,
              epoch: int, min_max_train: Tuple[float], recon_func: str) -> Tuple[float, float]:

    model.eval()
    labels_scores = []
    loss_function = model.get_loss_function(recon_func=recon_func)
    with torch.no_grad(), tqdm(total=len(anom_loader), desc='Eval anomalous') as pbar:
        for data, target in anom_loader:
            data = data.to(device)
            target = target.to(device)

            recon_batch, *distribution = model(data)

            # if min_max_train:
            #     anomaly_score, recon_err, kld_err = vae_loss_normalized(recon_batch, data, *distribution, min_max_train, recon_func=recon_func)
            # else:
            anomaly_score, recon_err, kld_err = loss_function(recon_batch, data, *distribution, min_max_train=min_max_train, frame_level=True)

            labels_scores += list(
                zip(target.view(-1).cpu().data.numpy().tolist(),
                    anomaly_score.view(-1).cpu().data.numpy().tolist(),
                    recon_err.view(-1).cpu().data.numpy().tolist(),
                    kld_err.reshape(-1).cpu().data.numpy().tolist(),
                    ))
            pbar.update()
        pbar.close()

        labels, anomaly_score_norm, recon_err_norm, kld_err_norm = zip(*labels_scores)

        roc_auc = roc_auc_score(labels, anomaly_score_norm)
        ap = average_precision_score(labels, anomaly_score_norm)

        roc_auc_recon = roc_auc_score(labels, recon_err_norm)
        ap_recon = average_precision_score(labels, recon_err_norm)

        roc_auc_kld = roc_auc_score(labels, kld_err_norm)
        ap_kld = average_precision_score(labels, kld_err_norm)

        if wandb.run:
            wandb.log({'roc_auc': roc_auc,
                       'ap': ap,
                       'roc_auc_recon': roc_auc_recon,
                       'ap_recon': ap_recon,
                       'roc_auc_kld': roc_auc_kld,
                       'ap_kld': ap_kld}, step=epoch)

        return roc_auc, ap


if __name__ == '__main__':
    import cv2
    model = ConVRNN(512, 512, 'elu').cuda()
    checkpoint = torch.load('/home/nello/expVAE/checkpoints/gradcam__conVRNN_03251226_chkp.pth')
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    anom = MovingMNIST(train=False, anom=True)
    anom_loader = DataLoader(anom, batch_size=64, shuffle=False, num_workers=4)

    with torch.no_grad(), tqdm(total=len(anom_loader), desc='Eval anomalous') as pbar:
        for i, (data, target) in enumerate(anom_loader, start=1):
            data = data.cuda()
            target = target.cuda()

            recon_batch, *distribution = model(data)
            for seq in range(data.shape[0]):
                for img in range(data.shape[2]):
                    if target[seq, img] == 1:
                        data[seq, :, img, 0, :] = 1
                        data[seq, :, img, -1, :] = 1
                        data[seq, :, img, :, 0] = 1
                        data[seq, :, img, :, -1] = 1

                diff = cv2.subtract(data[seq, :, :, :, :].cpu().numpy(), recon_batch[seq, :, :, :, :].cpu().numpy())
                diff[diff < 0.5] = 0

                imgs = gen_one_recon_img(data[seq], recon_batch[seq])
                imgs = torch.cat([imgs, torch.Tensor(diff).transpose(0, 1).cuda()], dim=0)

                save_image(imgs, f'./img_slides/ep-{epoch}_{(seq+1)*i}_recon_moving.png', nrow=20)
            pbar.update()
