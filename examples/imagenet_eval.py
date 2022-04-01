import argparse
import os
import sys
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ckpt_utils.load_func import latest_checkpoint_load
from optim import NKFAC, NAdam

import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/scratch/ssd002/datasets/imagenet_pytorch',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckpt-dir', type=str, default=None)

parser.add_argument('--optim', metavar='OPT', default='nkfac',
                    choices=['nkfac', 'nadam', 'sgd'], help='optimizer')
parser.add_argument('--ema-decay', type=float, default=0.99, metavar='EMA',
                    help='ema decay factor for cov update (default: 0.99)')
parser.add_argument('--kl-clip', type=float, default=1e-3, metavar='CLIP',
                    help='kl clipping constraint for trust region (default: 1e-3)')
parser.add_argument('--kl-lam', type=float, default=1.0, metavar='LAM',
                    help='kl weighting factor (default: 1.0)')
parser.add_argument('--precision', type=float, default=0.0, metavar='P',
                    help='prior precision (default: 1.0)')
parser.add_argument('--mul-factor', type=float, default=1.0, metavar='M',
                    help='scaling factor for noise sampling (default: 1.0)')
parser.add_argument('--mc-sample', type=int, default=1, metavar='MC',
                    help='number of mc samples used (default: 1)')
parser.add_argument('--num-bins', type=int, default=10, metavar='BIN',
                    help='number of bins used (default: 10)')


best_acc1 = 0
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

IMAGENET_SIZE = 1281167


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print("Using {} of GPUS".format(torch.cuda.device_count()))
    main_worker(args)


def main_worker(args):
    global best_acc1

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print(model)

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim == "nkfac":
        print("using Noisy-KFAC!")
        optimizer = NKFAC(model, dataset_size=IMAGENET_SIZE, lr=args.lr,
                          ema_decay=args.ema_decay, weight_decay=args.wd, kl_clip=args.kl_clip,
                          kl_lam=args.kl_lam, precision=args.precision, mul_factor=args.mul_factor)
    elif args.optim == "nadam":
        print("using Noisy-Adam!")
        optimizer = NAdam(model, dataset_size=IMAGENET_SIZE, lr=args.lr,
                          ema_decay=args.ema_decay, weight_decay=args.wd, kl_clip=args.kl_clip,
                          kl_lam=args.kl_lam, precision=args.precision, mul_factor=args.mul_factor)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.wd)

    #=======================================================================#
    if args.ckpt_dir is not None:
        checkpoint = latest_checkpoint_load(args.ckpt_dir)
    else:
        checkpoint = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint[0]['state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer'])
        print("Loaded a checkpoint\n")
    #=======================================================================#

    acc, conf, bin_sizes, bin_accs, bin_confs = validate(val_loader, model, optimizer, criterion, args)
    print("Acc: {} | Conf: {}".format(acc, conf))
    print(bin_sizes)
    print(bin_accs)
    print(bin_confs)


def validate(val_loader, model, optimizer, criterion, args):
    # switch to evaluate mode
    model.eval()

    num_bins = args.num_bins
    bins = np.linspace(1.0/num_bins, 1, num_bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    acc = 0.
    conf = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            prob = None
            for _ in range(args.mc_sample):
                if args.optim in ["nkfac", "nadam"] and args.mul_factor > 0.0:
                    optimizer.sample_params(args.mul_factor)

                output = model(images)
                if prob is None:
                    prob = F.softmax(output)
                else:
                    prob += F.softmax(output)
            prob /= args.mc_sample
            pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            conf += torch.gather(prob, dim=1, index=pred).sum().item()
            acc += pred.eq(target.view_as(pred)).sum().item()

            preds = prob.cpu().detach().numpy().flatten()
            label_onehot = F.one_hot(target, num_classes=1000)
            label_onehot = label_onehot.cpu().detach().numpy().flatten()
            assert preds.shape == label_onehot.shape
            for bin in range(num_bins):
                binned = np.digitize(preds, bins)

                if bin == 0:
                    print(len(preds[binned == bin]))
                bin_sizes[bin] += len(preds[binned == bin])
                bin_accs[bin] += (label_onehot[binned == bin]).sum()
                bin_confs[bin] += (preds[binned == bin]).sum()

    num_points = len(val_loader.dataset)
    return acc / num_points, conf / num_points, bin_sizes, bin_accs / (bin_sizes+1e-10), bin_confs / (bin_sizes+1e-10)



def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        prob = F.softmax(output)
        pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        confidence = torch.gather(prob, dim=1, index=pred).sum().item()
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct * 100.0 / batch_size, confidence * 100.0 / batch_size


if __name__ == '__main__':
    main()
