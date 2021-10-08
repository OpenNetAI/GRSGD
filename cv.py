import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse
import sys
import time
from functools import partial

import models
from utils import *

MODELS = {
    'resnet18': models.ResNet18,
    'resnet50': models.ResNet50,
    'desnet121': models.DenseNet121,
    'mobilenet': models.MobileNetV2,
    'googlenet': models.GoogLeNet,
    'CNNmnist':models.CNNmnist,
    'alexnet':models.AlexNet
}


# Training
def train(epoch, net, dataloader, criterion, optimizer, rank, args, writer):
    print('\nEpoch: %d' % epoch)
    net.train()

    OPTIMS = {
        'baseline': optimizer.step_base,
        's3sgd': optimizer.step_s3sgd,
        'topk': optimizer.step_topk,
        'mtopk': optimizer.step_mtopk,
        'dgc': optimizer.step_dgc,
        'tcs': optimizer.step_tcs
    }

    train_loss = 0
    correct = 0
    total = 0
    loader_len = len(dataloader)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if args.optim=='tcs' and epoch < 3:
            _, up_percent = OPTIMS['baseline']()
        else:
            _, up_percent = OPTIMS[args.optim]()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        step = loader_len * epoch + batch_idx
        writer.add_scalar(f'train_loss', train_loss / (batch_idx + 1), step)
        writer.add_scalar(f'train_acc', correct / total, step)
        writer.add_scalar(f'Up_percent', up_percent, step)
        print(f'Step [{epoch}-{step}] Training* loss:{train_loss / (batch_idx + 1)} | acc: {correct / total} | up_percent: {up_percent} | time: {time.time()}')

    return train_loss / loader_len, correct / total, correct, total


def test(epoch, net, dataloader, criterion, rank, writer, trainloader_len):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loader_len = len(dataloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f'Testing* loss:{test_loss / (batch_idx + 1)} | acc: {correct / total} | time: {time.time()}')
        step = trainloader_len * epoch
        writer.add_scalar(f'test_loss', test_loss / (batch_idx + 1), step)
        writer.add_scalar(f'test_acc', correct / total, step)
    return test_loss / loader_len, correct / total, correct, total


def main_worker(rank, args, gpus):
    sys.stdout = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)
    sys.stderr = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('gloo', world_size=args.world_size, rank=rank)
    gpu = gpus[rank % len(gpus)]
    torch.cuda.set_device(gpu)

    start_epoch = 0
    print(args)

    # Data
    print('==> Preparing data..')
    # =======================================数据集mnist=========================================================== #
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        sampler = torch.utils.data.DistributedSampler(dataset_train)
        trainloader = torch.utils.data.DataLoader(dataset_train,num_workers=4,batch_size=args.batch_size,shuffle=(sampler is None),sampler=sampler)
        dataset_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2,batch_size=args.batch_size,shuffle=False)
    # =======================================数据集cifar10=========================================================== #
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = torchvision.datasets.CIFAR10('../s3sgd_simu/data/cifar/', train=True, download=True, transform=transform_train)
        sampler = torch.utils.data.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=0, sampler=sampler)
        dataset_test = torchvision.datasets.CIFAR10('../s3sgd_simu/data/cifar/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=0)
    # =======================================数据集imagenet=========================================================== #
    elif args.dataset == 'imagenet':
        dataset_train = torchvision.datasets.ImageFolder(
            '/data/public/datasets/ImageNet/train',
            # 对数据进行预处理
            transforms.Compose([                      # 将几个transforms 组合在一起
                transforms.RandomSizedCrop(224),      # 随机切再resize成给定的size大小
                transforms.RandomHorizontalFlip(),    # 概率为0.5，随机水平翻转。
                transforms.ToTensor(),                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                                      # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        dataset_test = torchvision.datasets.ImageFolder('/data/public/datasets/ImageNet/val', transforms.Compose([
            # 重新改变大小为`size`，若：height>width`,则：(size*height/width, size)
            transforms.Scale(256),
            # 将给定的数据进行中心切割，得到给定的size。
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
        sampler = torch.utils.data.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=0, sampler=sampler)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=0)
        # exit(0)
    else:
        exit('Error: unrecognized dataset')

    # # copy weights and others to other model
    print('==> Building model..')
    net = MODELS[args.model]().cuda()
    for p in net.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'{args.checkpoint}/{rank:02}-ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print(f'==> Starting from {start_epoch}...')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = MultiStepLR(optimizer, [100, 200], gamma=0.1)

    writer = SummaryWriter(os.path.join(args.log_dir, f'rank_{rank:02}'), purge_step=start_epoch * len(trainloader))
    trainloader_len = len(trainloader)

    for epoch in range(start_epoch, args.epochs):
        loss, acc, correct, total = train(epoch, net, trainloader, criterion, optimizer, rank, args, writer)
        print('Rank: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %f' %
              (rank, loss, acc, correct, total, time.time()))
        if rank == 0 or args.test_all:
            loss, acc, correct, total = test(epoch, net, testloader, criterion, rank, writer, trainloader_len)
            print('Rank: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %f' %
                  (rank, loss, acc, correct, total, time.time()))
        sched.step(epoch)

        print('==>Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(state, f'./{args.checkpoint}/{rank:02}-ckpt.pth')

    # average weights
    for p in net.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

    if rank == 0 or args.test_all:
        loss, acc, correct, total = test(args.epochs, net, testloader, criterion, rank, writer, trainloader_len)
        print('Rank: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %f' %
              (rank, loss, acc, correct, total, time.time()))


def main():
    parser = argparse.ArgumentParser(description='s3sgd Simulation for CV')
    parser.add_argument('--model', default='CNNmnist', help='model name')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--master-addr', default='127.0.0.1', help='master addr')
    parser.add_argument('--master-port', default='25501', help='master port')
    parser.add_argument('--world-size', default=8, type=int,
                        help='node size in simulation')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=120, type=int, help="train epoch")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')  # TODO
    parser.add_argument('--download', default=False, action='store_true',
                        help="only download dataset")
    parser.add_argument('--data-dir', default='./data',
                        help='the data directory location')
    parser.add_argument('--gpus', required=True, help='gpu id the code runs on')
    parser.add_argument('--log-dir', default='./board/CNNmnist', help='train visual log location')
    parser.add_argument('--checkpoint', default='./checkpoint/CNNmnist', help='checkpoint location')
    parser.add_argument('--stdout', default='./stdout/CNNmnist', help='stdout log dir for subprocess')
    parser.add_argument('--test-all', default=False, action='store_true', help='run test on all nodes')
    parser.add_argument('--single', action='store_true', help='use with rank')
    parser.add_argument('--rank', default=-1, type=int, help='use with single')
    parser.add_argument('--optim', default='baseline', help='use which compressor')
    args = parser.parse_args()

    dirs = [args.data_dir, args.log_dir, args.checkpoint, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)

    gpus = [int(g) for g in args.gpus.split(',')]
    if args.single:
        main_worker(args.rank, args, gpus)
    else:
        mp.spawn(main_worker, args=(args, gpus), nprocs=args.world_size)


if __name__ == '__main__':
    main()
