from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb


def aggregate(model, train_loader: DataLoader,
              device: torch.device, recon_func: str) -> Tuple[Tensor, Tensor]:
    model.eval()
    latent_dim = model.latent_dim
    counter = 0
    mu_agg = torch.zeros(latent_dim).to(device)
    var_agg = torch.zeros(latent_dim).to(device)
    mu_squared_agg = torch.zeros(latent_dim).to(device)

    all_recon_errors, all_kld_errors = torch.Tensor(), torch.Tensor()

    loss_function = model.get_loss_function(recon_func=recon_func)

    with torch.no_grad(), tqdm(total=len(train_loader), desc='Aggregate') as pbar:
        for data, _ in train_loader:

            data = data.to(device)
            x_hat, *distribution = model(data)

            mu, logvar = distribution[0]
            recon_error, kld_error = loss_function(x_hat, data, *distribution, return_min_max=True)
            all_recon_errors = torch.cat((all_recon_errors, recon_error.view(-1).cpu()))
            all_kld_errors = torch.cat((all_kld_errors, kld_error.view(-1).cpu()))
            var = torch.exp(logvar)

            # var_agg += var.sum(dim=0)
            # mu_agg += mu.sum(dim=0)
            # mu_squared_agg += (mu ** 2).sum(dim=0)

            counter += len(data)
            pbar.update()
        pbar.close()

    mu_agg /= counter

    var_agg = var_agg / counter + mu_squared_agg / counter - mu_agg**2
    min_max_loss = torch.quantile(all_recon_errors, 0.03), torch.quantile(all_recon_errors, 0.97), \
        torch.quantile(all_kld_errors, 0.03), torch.quantile(all_kld_errors, 0.97)
    return mu_agg, var_agg, min_max_loss


def train(model, train_loader: DataLoader, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
          device: torch.device, epoch: int, recon_func: str, alpha: float = 1.0) -> None:
    total_loss = 0.0
    num_samples = 0
    loss_function = model.get_loss_function(recon_func=recon_func)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for data, _ in train_loader:

            model.train()
            data = data.to(device)
            batch_size = data.shape[0]

            num_samples += batch_size
            optimizer.zero_grad()

            x_recon, *distribution = model(data)

            loss, _, _ = loss_function(x_recon, data, *distribution, alpha)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix(loss=total_loss / num_samples,
                             alpha_kld=alpha.item(),
                             lr=optimizer.param_groups[0]['lr'])
            if wandb.run:
                wandb.log({'train_loss': total_loss / num_samples,
                           'alpha_kld': alpha.item(),
                           'lr': optimizer.param_groups[0]['lr']}, step=epoch)
        scheduler.step()
        pbar.close()


if __name__ == '__main__':
    from model.conVRNN import ConVRNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recon_func = 'bce'
    model = ConVRNN(512, 512, 'elu').to(device)
    from dataset.MovingMNIST import MovingMNIST
    train_moving = MovingMNIST(train=True)
    train_loader = DataLoader(train_moving, batch_size=16, shuffle=True, num_workers=4)

    aggregate(model, train_loader, device, recon_func)
