from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.vae_loss import vae_loss


def aggregate(model, train_loader: DataLoader,
              device: torch.device, recon_func: str) -> Tuple[Tensor, Tensor]:
    model.eval()
    latent_dim = model.latent_dim
    counter = 0
    mu_agg = torch.zeros(latent_dim).to(device)
    var_agg = torch.zeros(latent_dim).to(device)
    mu_squared_agg = torch.zeros(latent_dim).to(device)
    min_recon, min_kld = float('inf'), float('inf')
    max_recon, max_kld = float('-inf'), float('-inf')

    with torch.no_grad(), tqdm(total=len(train_loader), desc='Aggregate') as pbar:
        for data, _ in train_loader:

            data = data.to(device)
            x_hat, mu, logvar = model(data)

            _, min_max_loss = vae_loss(x_hat, data, mu, logvar, recon_func=recon_func, return_min_max=True)
            var = torch.exp(logvar)

            var_agg += var.sum(dim=0)
            mu_agg += mu.sum(dim=0)
            mu_squared_agg += (mu ** 2).sum(dim=0)

            counter += len(data)
            if min_max_loss[0] < min_recon:
                min_recon = min_max_loss[0]
            if min_max_loss[1] > max_recon:
                max_recon = min_max_loss[1]
            if min_max_loss[2] < min_kld:
                min_kld = min_max_loss[2]
            if min_max_loss[3] > max_kld:
                max_kld = min_max_loss[3]
            pbar.update()

    mu_agg /= counter

    var_agg = var_agg / counter + mu_squared_agg / counter - mu_agg**2
    min_max_loss = (min_recon, max_recon, min_kld, max_kld)
    return mu_agg, var_agg, min_max_loss


def train(model, train_loader: DataLoader, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
          device: torch.device, epoch: int, recon_func: str, alpha: float = 1.0) -> None:
    total_loss = 0.0
    num_samples = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for data, _ in train_loader:

            model.train()
            data = data.to(device)
            batch_size = data.shape[0]
            num_samples += batch_size
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            loss = vae_loss(x_recon, data, mu, logvar, recon_func, alpha)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix(loss=total_loss / num_samples,
                             alpha_kdl=alpha.item(),
                             lr=optimizer.param_groups[0]['lr'])
            if wandb.run:
                wandb.log({'train_loss': total_loss / num_samples,
                           'alpha_kdl': alpha.item(),
                           'lr': optimizer.param_groups[0]['lr']}, step=epoch)
        scheduler.step()
