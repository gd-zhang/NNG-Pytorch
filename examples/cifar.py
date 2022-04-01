import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import wandb
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import get_dataloader
from utils.kfac_utils import get_closure
from ckpt_utils.save_func import checkpoint_save
from ckpt_utils.load_func import latest_checkpoint_load
from optim import NKFAC, NAdam

import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        closure = get_closure(optimizer, model, data, target)
        loss, _ = optimizer.step(closure=closure)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, optimizer, device, test_loader, mc_sample=1, mul_factor=1.0):
    model.eval()
    correct = 0
    confidence = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            prob = None
            for _ in range(mc_sample):
                optimizer.sample_params(mul_factor)
                output = model(data)
                if prob is None:
                    prob = F.softmax(output)
                else:
                    prob += F.softmax(output)
            prob /= mc_sample
            pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            confidence += torch.gather(prob, dim=1, index=pred).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    conf = 100. * confidence / len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), acc))
    return acc, conf


def adjust_hparams(optimizer, epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 0.0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--ema-decay', type=float, default=0.99, metavar='EMA',
                        help='ema decay factor for cov update (default: 0.99)')
    parser.add_argument('--kl-lam', type=float, default=1.0, metavar='LAM',
                        help='kl weighting factor (default: 1.0)')
    parser.add_argument('--precision', type=float, default=0.0, metavar='P',
                        help='prior precision (default: 1.0)')
    parser.add_argument('--mul-factor', type=float, default=1.0, metavar='M',
                        help='scaling factor for noise sampling (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--arch', metavar='ARCH', default='vgg16_cifar',
                        choices=model_names, help='model architecture')
    parser.add_argument('--optim', metavar='OPT', default='kfac',
                        choices=['kfac', 'adam'], help='optimizer')
    # checkpointing
    parser.add_argument('--ckpt-dir', type=str, default=None)

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    if args.ckpt_dir is not None:
        wandb.init(project='NNG-CIFAR', config=state, save_code=True,
                   name='%s_%s_lr%.4f_wd%.5f_ema%.2f_lam%.2f_prec%.1f_mul%.2f' % (
                       args.arch, args.optim, args.lr, args.wd, args.ema_decay,
                       args.kl_lam, args.precision, args.mul_factor))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # init dataloader
    trainloader, testloader = get_dataloader(dataset="cifar10",
                                             train_batch_size=args.batch_size,
                                             test_batch_size=1000)

    model = models.__dict__[args.arch]()
    model = model.to(device)
    if args.optim == "kfac":
        optimizer = NKFAC(model, dataset_size=len(trainloader.dataset), lr=args.lr,
                          weight_decay=args.wd, ema_decay=args.ema_decay, mul_factor=args.mul_factor,
                          kl_clip=1e-3, kl_lam=args.kl_lam, precision=args.precision, gain=2.0)
    else:
        optimizer = NAdam(model, dataset_size=len(trainloader.dataset), lr=args.lr,
                          weight_decay=args.wd, ema_decay=args.ema_decay, mul_factor=args.mul_factor,
                          kl_clip=1e-3, kl_lam=args.kl_lam, precision=args.precision, gain=2.0)
    model = torch.nn.DataParallel(model)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    if args.ckpt_dir is not None:
        checkpoint = latest_checkpoint_load(args.ckpt_dir)
    else:
        checkpoint = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint[0]['state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer'])
        # scheduler.load_state_dict(checkpoint[0]['scheduler'])
        start_epoch = checkpoint[0]['epoch']
        print("Loaded a checkpoint\n")
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        begin_time = time.time()
        train(args, model, device, trainloader, optimizer, epoch)
        train_time = time.time() - begin_time
        print("Epoch: {} | Time elapsed: {:.1f}".format(epoch, train_time))

        # optimizer.sample_params()
        train_acc, train_conf = test(model, optimizer, device, trainloader, mc_sample=10, mul_factor=args.mul_factor)
        test_acc, test_conf = test(model, optimizer, device, testloader, mc_sample=10, mul_factor=args.mul_factor)
        # scheduler.step()
        adjust_hparams(optimizer, epoch, args.schedule)

        if args.ckpt_dir is not None:
            wandb.log({'train-acc': train_acc, 'train-conf': train_conf,
                       'test-acc': test_acc, 'test-conf': test_conf}, step=epoch)
            checkpoint_name = checkpoint_save({'state_dict': model.state_dict(),
                                               'optimizer': optimizer.state_dict(),
                                               # 'scheduler': scheduler.state_dict(),
                                               'epoch': epoch + 1}, args.ckpt_dir)

if __name__ == '__main__':
    main()
