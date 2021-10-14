import torch
import torch.nn as nn
import optimizers
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Batch
from mpl_toolkits.axisartist.axislines import SubplotZero
from tensorboardX import SummaryWriter

import spacy
import math
import argparse
import os
import sys

from models.seq2seq import *
from utils import *


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epoch, model, iterator, optimizer, criterion, clip, rank, args, writer):
    model.train()
    epoch_loss = 0
    loader_len = len(iterator)
    OPTIMS = {
        'baseline': optimizer.step_base,
        'grsgd': optimizer.step_grsgd,
        'topk': optimizer.step_topk,
        'mtopk': optimizer.step_mtopk,
        'tcs': optimizer.step_tcs,
        'dgc': optimizer.step_dgc,
    }

    for i, batch in enumerate(iterator):
        src = batch['src'].cuda()
        trg = batch['trg'].cuda()
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if args.optim=='tcs' and epoch < args.tcs_warmup_epochs:
            _, up_percent = OPTIMS['baseline']()
        else:
            _, up_percent = OPTIMS[args.optim]()

        epoch_loss += loss.item()
        step = epoch * loader_len + i
        writer.add_scalar(f'train_loss', epoch_loss / (i + 1), step)
        writer.add_scalar(f'train_ppl', math.exp(epoch_loss / (i + 1)), step)
        writer.add_scalar(f'Up_percent', up_percent, step)
        print(f'[{epoch}][{i}/{loader_len}]Training* loss:{epoch_loss/(i+1):.4f} | ppl: {math.exp(epoch_loss/(i+1)):.4f} | up_percent: {up_percent:.4f}')

    return epoch_loss / loader_len


def evaluate(epoch, model, iterator, criterion, rank, writer, trainloader):
    model.eval()

    epoch_loss = 0
    loader_len = len(iterator)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src'].cuda()
            trg = batch['trg'].cuda()

            output = model(src, trg, 0)  # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            step = epoch * loader_len + i
            if step%10 == 0:
                print(f'Testing* loss:{epoch_loss/(i+1)} | ppl: {math.exp(epoch_loss/(i+1))}')
        iteration = len(trainloader)*(epoch+1)
        writer.add_scalar(f'eval_loss', epoch_loss/loader_len, iteration)
        writer.add_scalar(f'eval_ppl', math.exp(epoch_loss/loader_len), iteration)

    return epoch_loss / loader_len


def build_group(rank: int, iter: int):
    return [(rank + iter)]


def main_work(rank, args, gpus):
    sys.stdout = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)
    sys.stderr = open(f'{args.stdout}/{rank:02}_stdout.log', 'a+', 1)

    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('gloo', world_size=args.world_size, rank=rank)
    gpu = gpus[rank % len(gpus)]
    torch.cuda.set_device(gpu)

    start_epoch = 0
    print(args)

    enc_emb_dim = args.enc_emb
    dec_emb_dim = args.dec_emb
    enc_hid_dim = args.enc_hid
    dec_hid_dim = args.dec_hid
    enc_dropout = args.enc_drop
    dec_dropout = args.dec_drop
    batch_size = args.batch_size
    n_epochs = args.epochs
    clip = 1

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # Tokenize text from a string into a list of strings in lambda
    source = Field(tokenize=lambda text: [tok.text for tok in spacy_de.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    target = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    train_data, valid_data, test_data = Multi30k.splits(root=args.data_dir,
                                                        exts=('.de', '.en'),
                                                        fields=(source, target))

    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)
    input_dim = len(source.vocab)
    output_dim = len(target.vocab)

    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    # average weights
    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'{args.checkpoint}/{rank:02}-ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print(f'==> Starting from {start_epoch}...')

    # TODO: should have a way to process data on gpu
    def torchtext_collate(data):
        b = Batch(data, train_data)
        return {'src': b.src, 'trg': b.trg}

    sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=torchtext_collate,
                              sampler=sampler, shuffle=False,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=torchtext_collate,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=torchtext_collate,
                             shuffle=False, num_workers=0, pin_memory=True)

    optimizer = optimizers.Adam(model.parameters())

    PAD_IDX = target.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    writer = SummaryWriter(os.path.join(args.log_dir, f'rank_{rank:02}'), purge_step=start_epoch * len(train_loader))
    print(f'Rank: {rank} starting on GPU: {gpu}...')
    for epoch in range(start_epoch, n_epochs):
        print(f'==>Epoch: {epoch}')
        train_loss = train(epoch, model, train_loader, optimizer, criterion, clip, rank, args, writer)
        print(f'\t Rank: {rank} | Train Loss: {train_loss:.3f} | '
              f'Train PPL: {math.exp(train_loss):7.3f}')
        if args.test_all or rank == 0:
            valid_loss = evaluate(epoch, model, valid_loader, criterion, rank, writer, train_loader)
            print(f'\t Rank: {rank} | Val. Loss: {valid_loss:.3f} |  '
                  f'Val. PPL: {math.exp(valid_loss):7.3f}')

        print('==>Saving...')
        state = {
            'net': model.state_dict(),
            'epoch': epoch+1
        }
        torch.save(state, f'./{args.checkpoint}/{rank:02}-ckpt.pth')

    # average weights
    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

    if args.test_all or rank == 0:
        test_loss = evaluate(n_epochs, model, test_loader, criterion, rank, writer)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')


def main():
    parser = argparse.ArgumentParser(description='GRSGD Simulation for Multi30k')
    parser.add_argument('--world-size', default=4, type=int,
                        help='node size in simulation')
    parser.add_argument('--master-addr', default='127.0.0.1', help='master addr')
    parser.add_argument('--master-port', default='25500', help='master port')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=20, type=int, help="train epoch")
    parser.add_argument('--rings', default=4, type=int,
                        help='the number of rings in world')
    parser.add_argument('--data-dir', default='./data',
                        help='the data directory location')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--log-dir', default='./board/test', help='train visual log location')
    parser.add_argument('--checkpoint', default='./checkpoint/test', help='checkpoint location')
    parser.add_argument('--stdout', default='./stdout/test', help='stdout log dir for subprocess')
    parser.add_argument('--gpus', required=True, help='gpu id the code runs on')
    parser.add_argument('--seed', default=12345, type=int, help="pytorch random seed")
    parser.add_argument('--test-all', default=False, action='store_true', help='run test on all nodes')

    parser.add_argument('--enc-emb', default=256, type=int, help='Encoder embedding size')
    parser.add_argument('--dec-emb', default=256, type=int, help='Decoder embedding size')
    parser.add_argument('--enc-hid', default=512, type=int, help='Encoder hidden layer size')
    parser.add_argument('--dec-hid', default=512, type=int, help='Decoder hidden layer size')
    parser.add_argument('--enc-drop', default=0.5, type=float, help='Encoder dropout probability ')
    parser.add_argument('--dec-drop', default=0.5, type=float, help='Decoder dropout probability ')
    parser.add_argument('--single', action='store_true', help='use with rank')
    parser.add_argument('--rank', default=-1, type=int, help='use with single')
    parser.add_argument('--optim', default='baseline', help='use which compressor')
    parser.add_argument('--tcs_warmup_epochs', default=2, type=int, help='tcs warmup epoch number')
    args = parser.parse_args()

    dirs = [args.data_dir, args.log_dir, args.checkpoint, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)

    gpus = [int(g) for g in args.gpus.split(',')]
    if args.single:
        main_work(args.rank, args, gpus)
    else:
        mp.spawn(main_work, args=(args, gpus), nprocs=args.world_size)


if __name__ == '__main__':
    main()
