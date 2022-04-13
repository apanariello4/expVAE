import argparse
import datetime

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.r2p1d import ResNet
from utils.dataset_loaders import load_moving_mnist_resnet
from utils.utils import save_checkpoint
from model.focal import FocalLoss


def train_resnet(model: ResNet, train_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, epoch: int) -> float:
    model.train()
    train_loss = .0
    running_accuracy = .0
    with tqdm(desc=f"[{epoch}] Train", total=len(train_loader)) as pbar:
        for i, (data, targets) in enumerate(train_loader):

            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            running_accuracy += ((outputs.argmax(dim=1) == targets).float()).sum().item()
            pbar.update()
            pbar.set_postfix(loss=train_loss / (i + 1),
                             lr=optimizer.param_groups[0]['lr'])

        pbar.close()
        acc = running_accuracy / len(train_loader.dataset)
        if wandb.run:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': acc,
            }, step=epoch)
    return train_loss, acc


def test_resnet(model: ResNet, test_loader: DataLoader,
                criterion: nn.Module, device: torch.device,
                epoch: int) -> float:
    model.eval()
    test_loss = .0
    scores = []
    labels = []
    running_accuracy = .0
    with tqdm(desc=f"[{epoch}] Test", total=len(test_loader)) as pbar:
        for i, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            scores += outputs[:, 1].detach().cpu().numpy().tolist()
            labels += targets.detach().cpu().numpy().tolist()
            running_accuracy += ((outputs.argmax(dim=1) == targets).float()).sum().item()
            pbar.update()
            pbar.set_postfix(loss=test_loss / (i + 1))
        pbar.close()
    acc = running_accuracy / len(test_loader.dataset)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    if wandb.run:
        wandb.log({
            'test_loss': test_loss,
            'test_acc': acc,
            'auc': auc,
            'ap': ap,
        }, step=epoch)

    return test_loss, acc, auc, ap


def main_resnet(model: ResNet, args: argparse.Namespace):
    now = datetime.datetime.now().strftime("%m%d%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, test_loader = load_moving_mnist_resnet(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    if args.log:
        name = args.name + '_' if args.name else ''
        wandb.init(project='resnet_moving', name=f"{name}{model.name}_{now}", config=args)
        wandb.watch(model)

    if args.test_only:
        test_loss, auc, ap = test_resnet(model, test_loader, criterion, device, epoch='test_only')
        print(f'Test loss: {test_loss:.4f}, Test ROC-AUC: {auc:.4f}, AP-AUC: {ap:.4f}')
        exit()

    best_test_loss = 1e10
    for epoch in range(args.epochs):
        train_loss, train_acc = train_resnet(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc, auc, ap = test_resnet(model, test_loader, criterion, device, epoch)
        print(f'[{epoch}] Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}\n',
              f'\tTest loss: {test_loss:.4f}, Test accuracy: {test_acc}\t ROC-AUC: {auc:.4f}, AP-AUC: {ap:.4f}')

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
