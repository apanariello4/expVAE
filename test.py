from typing import Tuple
import torch
import wandb
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.vae_loss import vae_loss, vae_loss_normalized
from sklearn.metrics import roc_auc_score


def save_one_recon_batch(model, device, test_loader, epoch):
    with torch.no_grad():
        for it, (x, _) in enumerate(test_loader):
            model.eval()
            x = x.to(device)

            x_hat, _, _ = model(x)

            imgs = torch.cat([x[0].transpose(0, 1), x_hat[0].transpose(0, 1)], dim=0)

            save_image(imgs, f'ep-{epoch}_recon_moving.png', nrow=10)

            break


def gen_one_recon_img(x: Tensor, x_hat: Tensor) -> Tensor:
    imgs = torch.cat([x.transpose(0, 1), x_hat.transpose(0, 1)], dim=0)
    return imgs


def eval(model, device: torch.device, test_loader: DataLoader, epoch: int) -> float:
    model.eval()
    test_loss = 0
    num_samples = 0
    with tqdm(total=len(test_loader), desc='Eval ') as pbar:
        with torch.no_grad():
            for data, _ in test_loader:

                data = data.to(device)

                batch_size = data.shape[0]
                num_samples += batch_size

                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                pbar.set_postfix(loss=test_loss / num_samples)
                pbar.update()

                imgs = gen_one_recon_img(data[0], recon_batch[0])

                if wandb.run:
                    w_imgs = wandb.Image(imgs, caption='Reconstruction')
                    wandb.log({'test_loss': test_loss / num_samples,
                               'reconstruction': w_imgs}, step=epoch)

    test_loss /= len(test_loader.dataset)
    return test_loss


def eval_anom(model, device: torch.device, anom_loader: DataLoader,
              epoch: int, min_max_train: Tuple[float]) -> float:

    model.eval()
    labels_scores = []
    with tqdm(total=len(anom_loader), desc='Eval anomalous') as pbar:
        with torch.no_grad():
            for data, target in anom_loader:
                data = data.to(device)
                target = target.to(device)

                recon_batch, mu, logvar = model(data)

                loss, scores = vae_loss_normalized(recon_batch, data, mu, logvar, min_max_train)
                scores = F.softmax(scores, dim=1)

                labels_scores += list(
                    zip(target.view(-1).cpu().data.numpy().tolist(),
                        scores.view(-1).cpu().data.numpy().tolist())
                )
                pbar.update()

            labels, scores = zip(*labels_scores)
            roc_auc = roc_auc_score(labels, scores)

            if wandb.run:
                wandb.log({'roc_auc': roc_auc}, step=epoch)

            return roc_auc


if __name__ == '__main__':
    from model.conv3dVAE import Conv3dVAE
    from utils.utils import args, load_moving_mnist
    arg = args()
    model = Conv3dVAE(latent_dim=512)
    checkpoint = torch.load('./checkpoints/moving_conv3dVAE_512_model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    arg.batch_size = 1
    _, test_loader = load_moving_mnist(arg)

    save_one_recon_batch(model, torch.device('cpu'), test_loader, checkpoint['epoch'])
