import argparse
from time import time
import math

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


best_acc1 = 0

epoch_loss = {}
epoch_loss['train'] = []
epoch_loss['val'] = []
epoch_acc = {}
epoch_acc['train'] = []
epoch_acc['val'] = []
epochs_list = []

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR quick training script')

    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset', default='./cifar-10-dataset')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100'],
                        default='cifar100')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float,
                        help='weight decay (default: 1e-4)')

    return parser


def main():
    global best_acc1
    global epochs_list

    parser = init_parser()
    args = parser.parse_args()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']

    from model2 import CCT
    model = CCT(img_size=img_size, embedding_dim=256, num_layers=7,
                num_heads=4, mlp_ratio=2, num_classes=num_classes)

    criterion = LabelSmoothingCrossEntropy()

    epochs_list = [i+1 for i in range(args.epochs)]

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model.cuda(0)
        criterion = criterion.cuda(0)

    base_optimizer = torch.optim.AdamW
    from sam import SAM
    optimizer = SAM(model.parameters(), base_optimizer,
                    lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0)

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = [CIFAR10Policy()]
    augmentations += [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    augmentations = transforms.Compose(augmentations)
    train_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=True, download=True, transform=augmentations)

    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    print("Beginning training")
    time_begin = time()
    for epoch in range(args.epochs):

        # adjust_learning_rate(optimizer, epoch, args)
        cls_train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = cls_validate(val_loader, model, criterion,
                            args, epoch=epoch, time_begin=time_begin)
        best_acc1 = max(acc1, best_acc1)
        scheduler.step()

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'final top-1: {acc1:.2f}')
    torch.save(model.state_dict(), args.checkpoint_path)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(epochs_list, epoch_loss['train'], label='train')
    ax.plot(epochs_list, epoch_loss['val'], label='val')
    ax.set_title("Loss Epoch Graph")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig('./epoch_loss.png')
    fig.show()
    # -----------------------------------
    fig, ax = plt.subplots()
    ax.plot(epochs_list, epoch_acc['train'], label='train')
    ax.plot(epochs_list, epoch_acc['val'], label='val')
    ax.set_title("Accuracy Epoch Graph")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig('./epoch_acc.png')
    fig.show()


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 20 and epoch < 30:
        lr = 0.001
    elif epoch >= 30 and epoch < 35:
        lr = 0.0006
    elif epoch >= 35:
        lr = 0.0003
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return None

    tot_epochs = args.epochs
    LR = args.lr
    stepsize = 2
    k_min = 0.1
    k_max = 0.08

    N_MIN = LR*0.9
    N_MAX = LR

    n_min = N_MIN
    n_max = N_MAX

    lr = LR

    for epoch in range(1, tot_epochs+1):
        warmup = args.warmup
        if epoch <= warmup:
            lr = lr
        else:
            ep = epoch-warmup
            n_max = N_MAX*math.exp(-ep*k_max)
            n_min = N_MIN*math.exp(-ep*k_min)
            cycle = 1+ep//(2*stepsize)
            x = abs(ep/stepsize-2*cycle+1)
            lr = n_min+(n_max-n_min)*max(0, 1-x)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

        def closure():
            loss = criterion(model(images), target)
            loss.backward()
            return loss

        output = model(images)

        loss = criterion(output, target)

        acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()

        # if args.clip_grad_norm > 0:
        #     nn.utils.clip_grad_norm_(
        #         model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            print(
                f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    epoch_loss['train'].append(loss_val/n)
    epoch_acc['train'].append(acc1_val/n)


def cls_validate(val_loader, model, criterion, args, epoch=None, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                print(
                    f'[Epoch {epoch + 1}][Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    epoch_loss['val'].append(avg_loss)
    epoch_acc['val'].append(avg_acc1)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(
        f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc1


if __name__ == '__main__':
    main()
