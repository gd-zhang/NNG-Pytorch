import argparse
import os
import sys
import random
import time
import warnings
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ckpt_utils.save_func import checkpoint_save
from ckpt_utils.load_func import latest_checkpoint_load
from utils.kfac_utils import get_closure
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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--label-smoothing', default=0.0, type=float,
                    metavar='LS', help='label smoothing coeff')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--optim', metavar='OPT', default='nkfac',
                    choices=['nkfac', 'nadam', 'sgd'], help='optimizer')
parser.add_argument('--cosine-lr', default=False, action='store_true',
                    help='whether to use cosine annealing schedule')
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
parser.add_argument('--ckpt-dir', type=str, default=None)

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


best_acc1 = 0
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.ckpt_dir is not None:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if args.optim in ['nkfac', 'nadam']:
                name = '%s_%s_lr%.5f_wd%.5f_ema%.2f_clip%.5f_lam%.2f_prec%.1f_mul%.2f_ls%.2f' % (
                    args.arch, args.optim, args.lr, args.wd, args.ema_decay, args.kl_clip,
                    args.kl_lam, args.precision, args.mul_factor, args.label_smoothing)
            else:
                name='%s_%s_lr%.4f_wd%.5f_ls%.2f' % (args.arch, args.optim, args.lr, args.wd, args.label_smoothing)

            if args.cosine_lr:
                name += '_cosine'

            save_dir = os.path.join('results', name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            wandb.init(project='NNG-Imagenet', config=state, save_code=True,
                       name=name, dir=save_dir, resume=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print("Size of dataset: {}".format(len(train_dataset)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=(train_sampler is None))

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    if args.optim == "nkfac":
        print("using Noisy-KFAC!")
        optimizer = NKFAC(model, dataset_size=len(train_loader.dataset), lr=args.lr,
                          ema_decay=args.ema_decay, weight_decay=args.wd, kl_clip=args.kl_clip,
                          kl_lam=args.kl_lam, precision=args.precision, mul_factor=args.mul_factor)
    elif args.optim == "nadam":
        print("using Noisy-Adam!")
        optimizer = NAdam(model, dataset_size=len(train_loader.dataset), lr=args.lr,
                          ema_decay=args.ema_decay, weight_decay=args.wd, kl_clip=args.kl_clip,
                          kl_lam=args.kl_lam, precision=args.precision, mul_factor=args.mul_factor)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.wd)

    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.001)

    #=======================================================================#
    if args.ckpt_dir is not None:
        checkpoint = latest_checkpoint_load(args.ckpt_dir)
    else:
        checkpoint = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint[0]['state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer'])
        start_epoch = checkpoint[0]['epoch']
        if args.cosine_lr:
            scheduler.load_state_dict(checkpoint[0]['scheduler'])
        print("Loaded a checkpoint\n")
        del checkpoint
    else:
        start_epoch = 0
    #=======================================================================#

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, ngpus_per_node, args)

        # evaluate on validation set
        test_loss, test_acc, test_conf = validate(val_loader, model, optimizer, criterion, args)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(test_acc, best_acc1)

        # copy mean
        if args.optim in ["nkfac", "nadam"]:
            optimizer.copy_mean_to_params()

        # adjusting learning rate
        if args.cosine_lr:
            scheduler.step()
            curr_lr = scheduler.get_lr()[0]
        else:
            curr_lr = adjust_learning_rate(optimizer, epoch, args)

        print('\n====>epoch: %d' % (epoch))
        print('    train acc: %.4f, train loss: %.5f' % (train_acc, train_loss))
        print('    test acc: %.4f, test conf: %.2f, test loss: %.5f' % (test_acc, test_conf, test_loss))

        if args.ckpt_dir is not None:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                wandb.log({'train-acc': train_acc, 'train-loss': train_loss, 'lr': curr_lr,
                           'test-acc': test_acc, 'test-conf': test_conf, 'test-loss': test_loss},
                          step=(epoch+1)*len(train_loader))
                checkpoint_name = checkpoint_save({'state_dict': model.state_dict(),
                                                   'optimizer': optimizer.state_dict(),
                                                   'epoch': epoch+1,
                                                   'scheduler': scheduler.state_dict() if args.cosine_lr else None},
                                                  args.ckpt_dir)


def train(train_loader, model, criterion, optimizer, epoch, ngpus_per_node, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.optim in ["nkfac", "nadam"]:
            closure = get_closure(optimizer, model, images, target, criterion)
            loss, output = optimizer.step(closure=closure)
        else:
            # compute output
            output = model(images)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        acc, _ = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                wandb.log({"stats/time": batch_time.avg, "stats/data-time": data_time.avg},
                          step=epoch*len(train_loader) + i + 1)
                progress.display(i)

    return (losses.avg, top1.avg)


def validate(val_loader, model, optimizer, criterion, args, mean_pred=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc1 = AverageMeter('Acc', ':6.2f')
    confidence = AverageMeter('Conf', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc1, confidence],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            prob = None
            mc_sample = 1 if mean_pred else args.mc_sample
            for _ in range(mc_sample):
                if args.optim in ["nkfac", "nadam"] and args.mul_factor > 0.0 and not mean_pred:
                    optimizer.sample_params(args.mul_factor)

                output = model(images)
                if prob is None:
                    prob = F.softmax(output)
                else:
                    prob += F.softmax(output)
            prob /= args.mc_sample
            pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            conf = torch.gather(prob, dim=1, index=pred).sum().item()
            acc = pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc, conf = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            acc1.update(acc * 100.0 / images.size(0), images.size(0))
            confidence.update(conf * 100.0 / images.size(0), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {acc1.avg:.3f} Conf {confidence.avg:.3f}'
              .format(acc1=acc1, confidence=confidence))

    return (losses.avg, acc1.avg, confidence.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
