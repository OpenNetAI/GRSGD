import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter

def main():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                        help='warmup epoch number')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:25500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    # parser.add_argument('--dist-rank', default=0, type=int,
    #                     help='rank of distributed processes')
    parser.add_argument('--gpus', required=True, help='gpu id the code runs on')
    parser.add_argument('--checkpoint', default='./checkpoint/CNNmnist', help='checkpoint location')
    parser.add_argument('--stdout', default='./stdout/CNNmnist')
    parser.add_argument('--log-dir', default='./board/CNNmnist', help='train visual log location')
    parser.add_argument('--master-addr', default='127.0.0.1', help='master addr')
    parser.add_argument('--master-port', default='25501', help='master port')
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--optim', default='baseline', help='use which compressor')
    args = parser.parse_args()

    dirs = [args.log_dir, args.checkpoint, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)

    best_prec1 = 0
    # print(models.__dict__)
    # exit(0)

    gpus = [int(g) for g in args.gpus.split(',')]
    mp.spawn(main_worker, args=(args, gpus, best_prec1), nprocs=args.world_size)

  
def main_worker(rank, args, gpus, best_prec1):
    sys.stdout = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)
    sys.stderr = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,rank=rank)
    gpu = gpus[rank % len(gpus)]
    torch.cuda.set_device(gpu)

    print(args)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))

        model = models.__dict__[args.arch]()

        model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[gpu], output_device=gpu)

    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        print("=> loading checkpoint '{}'".format(f'{args.checkpoint}/{rank:02}-ckpt.pth'))
        checkpoint = torch.load(f'{args.checkpoint}/{rank:02}-ckpt.pth')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'==> Starting from {args.start_epoch}...')

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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    writer = SummaryWriter(os.path.join(args.log_dir, f'rank_{rank:02}'),
                           purge_step=args.start_epoch * len(train_loader))
    trainloader_len = len(train_loader)
    if args.evaluate:
        validate(writer, args, val_loader, model, criterion, epoch=-1, trainloader_len=trainloader_len)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(writer, args, train_loader, model, criterion, optimizer, epoch)


        # evaluate on validation set
        prec1 = validate(writer, args, val_loader, model, criterion, epoch, trainloader_len)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, f'./{args.checkpoint}/{rank:02}-ckpt.pth')


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < args.warmup_epochs:
        lr = args.lr / args.warmup_epochs * (epoch+1)
    else:
        lr = args.lr * (0.1 ** (epoch // 100))
    print('here:',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(writer, args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    OPTIMS = {
        'baseline': optimizer.step_base,
        's3sgd': optimizer.step_s3sgd,
        'topk': optimizer.step_topk
    }

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # optimizer.step()
        _, up_percent = OPTIMS[args.optim]()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step = len(train_loader) * epoch + i

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Up_percent {up_percent:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, up_percent=up_percent))
            writer.add_scalar(f'train_loss', losses.val, step)
            writer.add_scalar(f'train_acc', top1.val, step)
            writer.add_scalar(f'Up_percent', up_percent, step)


def validate(writer, args, val_loader, model, criterion, epoch, trainloader_len):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            step = trainloader_len * epoch

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        writer.add_scalar(f'test_loss', losses.val, step)
        writer.add_scalar(f'test_acc', top1.val, step)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
