from typing import Optional, Tuple
from model.vae_loss import vae_loss
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


def aggregate(model, train_loader: DataLoader,
              device: torch.device) -> Tuple[Tensor, Tensor]:
    model.eval()
    latent_dim = model.latent_dim
    counter = 0
    mu_agg = torch.zeros(latent_dim).to(device)
    var_agg = torch.zeros(latent_dim).to(device)
    mu_squared_agg = torch.zeros(latent_dim).to(device)

    with torch.no_grad():
        with tqdm(total=len(train_loader), desc='Aggregate') as pbar:
            for it, (data, _) in enumerate(train_loader):

                data = data.to(device)
                _, mu, logvar = model(data)

                var = torch.exp(logvar)

                var_agg += var.sum(dim=0)
                mu_agg += mu.sum(dim=0)
                mu_squared_agg += (mu ** 2).sum(dim=0)

                counter += len(data)
                pbar.update()

    mu_agg /= counter

    var_agg = var_agg / counter + mu_squared_agg / counter - mu_agg**2

    return mu_agg, var_agg


def train(model, train_loader: DataLoader, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
          device: torch.device, epoch: int, alpha: float = 1.0) -> None:
    total_loss = 0.0
    num_samples = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}') as pbar:
        for data, _ in train_loader:

            model.train()
            data = data.to(device)
            batch_size = data.shape[0]
            num_samples += batch_size
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            loss = vae_loss(
                x_recon=x_recon,
                x=data,
                mu=mu,
                logvar=logvar,
                alpha=alpha)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix(loss=total_loss / num_samples,
                             alpha_kdl=alpha.item(),
                             lr=optimizer.param_groups[0]['lr'])
        scheduler.step()
