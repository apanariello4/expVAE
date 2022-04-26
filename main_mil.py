import argparse
import datetime
from sched import scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.base_model import BaseModel
from model.mil_loss import MIL_loss, RegularizedMIL
from model.vae_loss import kld_gauss, nll_bernoulli
from utils.dataset_loaders import load_moving_mnist_mil
from utils.utils import get_alpha_scheduler, get_scheduler, save_checkpoint


def train_mil(model: BaseModel, train_loader: DataLoader,
              criterion: nn.Module, optimizer: torch.optim.Optimizer,
              scheduler, device: torch.device, epoch: int, alpha: float = 1.) -> float:
    model.train()
    train_loss = .0
    with tqdm(desc=f"[{epoch}] Train", total=len(train_loader)) as pbar:
        for i, ((norm, anom), frame_level_labels) in enumerate(train_loader):
            targets = torch.tensor([0] * len(norm) + [1] * len(anom), dtype=torch.float, device=device)
            data = torch.cat([norm, anom], dim=0)
            data = data.to(device)

            x_recon, *distribution, labels = model(data)

            nll = nll_bernoulli(x_recon, data, seq_level=True)
            masked_nll = nll * (1 - targets)
            kld = kld_gauss(*distribution[0], *distribution[1])
            mil = criterion(labels, targets)

            loss = mil + (masked_nll.sum() + alpha * kld) / data.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.update()
            pbar.set_postfix(last_batch_loss=loss.item(),
                             lr=optimizer.param_groups[0]['lr'],
                             mil=mil.item(),
                             )
        scheduler.step()
        pbar.close()
        train_loss /= len(train_loader)
        if wandb.run:
            wandb.log({
                'train_loss': train_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'alpha_kl': alpha,
            }, step=epoch)
    return train_loss


def test_mil(model: BaseModel, test_loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             epoch: int) -> float:
    model.eval()
    test_loss = .0
    roc_scores_seq = []
    roc_labels_seq = []
    roc_labels_frame = []
    roc_scores_frame = []
    roc_scores_classifier_seq = []
    roc_scores_classifier_frame = []
    with torch.no_grad(), tqdm(desc=f"[{epoch}] Test", total=len(test_loader)) as pbar:
        for i, ((norm, anom), frame_level_labels) in enumerate(test_loader):
            frame_level_labels = torch.cat([torch.zeros_like(frame_level_labels), frame_level_labels], dim=0)
            data = torch.cat([norm, anom], dim=0)
            data = data.to(device)
            targets = torch.tensor([0] * len(norm) + [1] * len(anom), dtype=torch.float, device=device)
            x_recon, *distribution, labels = model(data)
            nll = nll_bernoulli(x_recon, data, seq_level=True)
            masked_nll = nll * (1 - targets)
            mil = criterion(labels, targets)
            kld = kld_gauss(*distribution[0], *distribution[1])

            loss = mil + (masked_nll.sum() + kld) / data.shape[0]

            test_loss += loss.item()

            roc_scores_classifier_seq += labels.max(dim=1)[0].tolist()
            roc_scores_classifier_frame += labels.reshape(-1).cpu().numpy().tolist()

            scores_seq_level = nll_bernoulli(x_recon, data, seq_level=True)
            roc_scores_seq += scores_seq_level.detach().cpu().numpy().tolist()
            roc_labels_seq += targets.detach().cpu().numpy().tolist()

            scores_frame_level = nll_bernoulli(x_recon, data, frame_level=True)
            roc_scores_frame += scores_frame_level.view(-1).detach().cpu().numpy().tolist()
            roc_labels_frame += frame_level_labels.view(-1).detach().cpu().numpy().tolist()

            pbar.set_postfix(last_batch_loss=loss.item())
            pbar.update()

        pbar.close()
    test_loss /= len(test_loader)

    auc = roc_auc_score(roc_labels_seq, roc_scores_seq)
    roc_frame = roc_auc_score(roc_labels_frame, roc_scores_frame)
    roc_classifier_seq = roc_auc_score(roc_labels_seq, roc_scores_classifier_seq)
    roc_classifier_frame = roc_auc_score(roc_labels_frame, roc_scores_classifier_frame)

    ap = average_precision_score(roc_labels_seq, roc_scores_seq)
    ap_frame = average_precision_score(roc_labels_frame, roc_scores_frame)
    ap_classifier_seq = average_precision_score(roc_labels_seq, roc_scores_classifier_seq)
    ap_classifier_frame = average_precision_score(roc_labels_frame, roc_scores_classifier_frame)

    seq_norm = model.sample(anom=False)
    seq_anom = model.sample(anom=True)

    if wandb.run:
        wandb.log({
            'test_loss': test_loss,
            'auc': auc,
            'ap': ap,
            'roc_frame': roc_frame,
            'ap_frame': ap_frame,
            'roc_classifier': roc_classifier_seq,
            'ap_classifier': ap_classifier_seq,
            'roc_classifier_frame': roc_classifier_frame,
            'ap_classifier_frame': ap_classifier_frame,
            'seq_norm': wandb.Image(seq_norm, caption=f'Normal Sample ep_{epoch}'),
            'seq_anom': wandb.Image(seq_anom, caption=f'Anomalous Sample ep_{epoch}'),
        }, step=epoch, commit=True)

    return test_loss, auc, ap, roc_frame, ap_frame, roc_classifier_seq, ap_classifier_seq, roc_classifier_frame, ap_classifier_frame


def main_mil(model: BaseModel, args: argparse.Namespace):
    num_params = sum(p.numel() for p in model.parameters())
    num_params_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model.name} - Params: Total {num_params:,}, learnable {num_params_learn:,}')
    now = datetime.datetime.now().strftime("%m%d%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Loading dataset...')
    train_loader, test_loader = load_moving_mnist_mil(args)
    criterion = RegularizedMIL(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    scheduler = get_scheduler(optimizer, args)
    alpha = get_alpha_scheduler(args)

    if args.log:
        name = args.name + '_' if args.name else ''
        wandb.init(project='vrnn-mil', name=f"{name}{model.name}_{now}", config=args)
        wandb.watch(model)

    print(f"No skill classifier AP on frames: {test_loader.dataset.n_anom_frames/test_loader.dataset.n_frames}")

    if args.test_only:
        test_loss, auc, ap, roc_frame, ap_frame, roc_classifier_seq, ap_classifier_seq, roc_classifier_frame, ap_classifier_frame = test_mil(model, test_loader, criterion, device, epoch='test_only')
        print(f'Test loss: {test_loss:.4f}, Test ROC-AUC: {auc:.4f}, AP-AUC: {ap:.4f}')
        exit()

    best_test_loss = 1e10
    for epoch in range(args.epochs):
        train_loss = train_mil(model, train_loader, criterion, optimizer, scheduler, device, epoch, alpha[epoch])
        test_loss, auc, ap, roc_frame, ap_frame, roc_classifier_seq, ap_classifier_seq, roc_classifier_frame, ap_classifier_frame = test_mil(model, test_loader, criterion, device, epoch)
        print(f'[{epoch}]\tTrain loss: {train_loss:.4f}\t Test loss: {test_loss:.4f}\n',
              f'\tSequence:       ROC-AUC: {auc:.4f}, AP-AUC: {ap:.4f}',
              f'    Frame:            ROC-AUC: {roc_frame:.4f}, AP-AUC: {ap_frame:.4f}\n',
              f'\tClassifier Seq: ROC-AUC: {roc_classifier_seq:.4f}, AP-AUC: {ap_classifier_seq:.4f}',
              f'    Classifier Frame: ROC-AUC: {roc_classifier_frame:.4f}, AP-AUC: {ap_classifier_frame:.4f}')

        if args.save_checkpoint:
            if test_loss < best_test_loss or epoch == args.epochs - 1:
                best_test_loss = test_loss
                save_checkpoint(
                    model,
                    epoch,
                    optimizer,  # (mu_avg, var_avg),
                    is_last=epoch == args.epochs - 1,
                    args=args, time=now)
    exit()
