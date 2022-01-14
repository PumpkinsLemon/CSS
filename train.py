from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import resnet_css

pdist = nn.PairwiseDistance(p=2)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Model
parser.add_argument('--model_t', metavar='ARCH', default='resnet110', type=str, 
                    help='model architecture (default: resnet110)')    
parser.add_argument('--model_s', metavar='ARCH', default='resnet56', type=str,
                    help='model architecture (default: resnet56)')    
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# CSS parameter
parser.add_argument('--branch', type=int, default=3, help='the number of branch')
parser.add_argument('-T', '--temperature', type=float, default=3.0, help='the temperature of distillation loss')
parser.add_argument('--get_dis', action='store_true', help='Decide whether or not to calculate diversity: default(False)')
# Device options
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state_t = {k: v for k, v in args._get_kwargs()}
state_s = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc_t = 0
best_acc_s = 0
best_epoch_t = 0
best_epoch_s = 0
best_acc = 0
best_epoch = 0


def main():
    global best_acc_t, best_acc_s, best_acc, best_epoch_t, best_epoch_s, best_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset %s' % args.dataset)
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

    global num_classes
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        path = "xxx"  # cifar10 dataset location
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        path = "xxx"  # cifar100 dataset location
    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model (Teacher & Student)")

    model_t = getattr(resnet_css, args.model_t)(num_classes=num_classes, branch=args.branch)
    model_s = getattr(resnet_css, args.model_s)(num_classes=num_classes, branch=args.branch)
    model_t = torch.nn.DataParallel(model_t).cuda()
    model_s = torch.nn.DataParallel(model_s).cuda()
    cudnn.benchmark = True
    print('Teacher: %15s   Total params: %.2fM   ' % (args.model_t, sum(p.numel() for p in model_t.parameters())/1000000.0))
    print('Student: %15s   Total params: %.2fM   ' % (args.model_s, sum(p.numel() for p in model_s.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer_t = optim.SGD(model_t.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_s = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate_warmup(state_t, optimizer_t, epoch + 1)
        adjust_learning_rate_warmup(state_s, optimizer_s, epoch + 1)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state_t['lr']))

        train_acc_t, train_acc_s = train(trainloader, model_t, model_s, criterion, optimizer_t, optimizer_s, use_cuda, epoch)
        test_acc_t, test_acc_s, acc = test(testloader, model_t, model_s, criterion, epoch, use_cuda)

        is_best_t = test_acc_t > best_acc_t
        best_acc_t = max(test_acc_t, best_acc_t)
        if is_best_t is True:
            best_epoch_t = epoch + 1
        is_best_s = test_acc_s > best_acc_s
        best_acc_s = max(test_acc_s, best_acc_s)
        if is_best_s is True:
            best_epoch_s = epoch + 1
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best is True:
            best_epoch = epoch + 1
        print("Best Teacher    accuracy: {}".format(best_acc_t))
        print("Best Student    accuracy: {}".format(best_acc_s))
        print("Best Aggregated accuracy: {}".format(best_acc))

    print('------------------------------------------------------')
    print('Best Teacher    acc: %.2f  Best Teacher     Epoch: %d' % (best_acc_t, best_epoch_t))
    print('Best Student    acc: %.2f  Best Student     Epoch: %d' % (best_acc_s, best_epoch_s))
    print('Best Aggregated acc: %.2f  Best Aggregated  Epoch: %d' % (best_acc, best_epoch))


def train(trainloader, model_t, model_s, criterion, optimizer_t, optimizer_s, use_cuda, epoch):
    # switch to train mode
    model_t.train()
    model_s.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_t = AverageMeter()
    losses_s = AverageMeter()
    t_top1 = AverageMeter()
    s_top1 = AverageMeter()
    end = time.time()

    n = args.branch
    T = args.temperature

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # target diversity
        single_targets = []
        for i in range(n):
            single_targets.append(targets * n + i)
        # compute teacher output
        out_t = model_t(inputs)

        agg_out_t = 0
        for i in range(n):
            agg_out_t = agg_out_t + out_t[:, :, i][:,i::n] / n
        # compute student output
        out_s = model_s(inputs)
        agg_out_s = 0
        for i in range(n):
            agg_out_s = agg_out_s + out_s[:, :, i][:,i::n] / n 
    
        # Cross Entropy
        loss_ce = {'t':0, 's':0}
        for i in range(n):
            loss_ce['t'] += (criterion(out_t[:, :, i], single_targets[i])/n)
            loss_ce['s'] += (criterion(out_s[:, :, i], single_targets[i])/n)

        # KD loss
        loss_kd = {'t2s':0, 's2t':0}
        for i in range(n):
            loss_kd['t2s'] += (F.kl_div(F.log_softmax(out_t[:, :, i] / T, 1),
                                        F.softmax(out_s[:, :, i] / T, 1),
                                        reduction='batchmean')) / n
            loss_kd['s2t'] += (F.kl_div(F.log_softmax(out_s[:, :, i] / T, 1),
                                        F.softmax(out_t[:, :, i] / T, 1),
                                        reduction='batchmean')) / n
        # relative utility score
        alpha_1 = (loss_ce['t'] / loss_ce['s'])
        alpha_2 = (loss_ce['s'] / loss_ce['t'])
        
        loss_t = loss_ce['t'] + alpha_1 * loss_kd['t2s'].mul(T ** 2)
        loss_s = loss_ce['s'] + alpha_2 * loss_kd['s2t'].mul(T ** 2)
        loss = loss_s + loss_t

        # compute gradient and do SGD step
        optimizer_t.zero_grad()
        optimizer_s.zero_grad()
        loss.backward()
        optimizer_t.step()
        optimizer_s.step()

        t_prec1, _= accuracy(agg_out_t.data, targets.data, topk=(1, 5))
        s_prec1, _= accuracy(agg_out_s.data, targets.data, topk=(1, 5))

        # measure accuracy and record loss
        losses_t.update(loss_t.item(), inputs.size(0))
        losses_s.update(loss_s.item(), inputs.size(0))
        t_top1.update(t_prec1.item(), inputs.size(0))
        s_top1.update(s_prec1.item(), inputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % 20 == 0:
            print('\rTraining | Epoch: {}/{} | Batch: {}/{}| Losses_T: {:.4f} | T_Top-1: {:.2f} | Losses_S: {:.4f} | S_Top-1: {:.2f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, len(trainloader), losses_t.avg, t_top1.avg, losses_s.avg, s_top1.avg),
                end='', flush=True)
    print('\rTraining | Epoch: {}/{} | Batch: {}/{}| Losses_T: {:.4f} | T_Top-1: {:.2f} | Losses_S: {:.4f} | S_Top-1: {:.2f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, len(trainloader), losses_t.avg, t_top1.avg, losses_s.avg, s_top1.avg),
                end='\n')
    return (t_top1.avg, s_top1.avg)


@torch.no_grad()
def test(testloader, model_t, model_s, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_t = AverageMeter()
    losses_s = AverageMeter()
    t_top1 = AverageMeter()
    s_top1 = AverageMeter()
    agg_top1 = AverageMeter()

    # switch to evaluate mode
    model_t.eval()
    model_s.eval()

    n = args.branch
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute teacher output
        out_t = model_t(inputs)
        agg_out_t = 0
        for i in range(n):
            agg_out_t = agg_out_t + out_t[:, :, i][:,i::n] / n

        # compute student output
        out_s = model_s(inputs)
        agg_out_s = 0
        for i in range(n):
            agg_out_s = agg_out_s + out_s[:, :, i][:,i::n] / n
        
        agg_out = (agg_out_t + agg_out_s) / 2

        t_prec1, _= accuracy(agg_out_t.data, targets.data, topk=(1, 5))
        s_prec1, _= accuracy(agg_out_s.data, targets.data, topk=(1, 5))
        agg_prec1, _ = accuracy(agg_out.data, targets.data, topk=(1, 5))

        # measure accuracy and record loss
        t_top1.update(t_prec1.item(), inputs.size(0))
        s_top1.update(s_prec1.item(), inputs.size(0))
        agg_top1.update(agg_prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.get_dis:
            dist_all, dist_mutual = get_diversity(out_t, out_s, targets.size(0))

        if (batch_idx+1) % 20 == 0:
            print('\rTesting | Epoch: {}/{} | Batch: {}/{}| Losses_t: {:.4f} | T_Top-1: {:.2f} | Losses_s: {:.4f} | S_Top-1: {:.2f} | Agg_Top-1: {:.2f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, len(testloader), losses_t.avg, t_top1.avg, losses_s.avg, s_top1.avg, agg_top1.avg),
                end='', flush=True)
    print('\rTesting | Epoch: {}/{} | Batch: {}/{}| Losses_t: {:.4f} | T_Top-1: {:.2f} | Losses_s: {:.4f} | S_Top-1: {:.2f} | Agg_Top-1: {:.2f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, len(testloader), losses_t.avg, t_top1.avg, losses_s.avg, s_top1.avg, agg_top1.avg),
                end='\n')
    return (t_top1.avg, s_top1.avg, agg_top1.avg)


def get_diversity(out_t, out_s, length):

    dist_all = AverageMeter()
    dist_mutual = AverageMeter()

    n = args.branches

    # Computer L2 Distance
    '''
    Reference: AAAI20-Online Knowledge Distillation with Diverse Peers
    '''
    out_t = F.softmax(out_t, dim=1)
    out_s = F.softmax(out_s, dim=1)
    out = torch.cat((out_t, out_s), dim=-1)
    for i in range(length):
        ret = out[i,:,:]
        ret1 = out_t[i,:,:]
        ret2 = out_s[i,:,:]
        ret = ret.t()
        ret1 = ret1.t()
        ret2 = ret2.t()
        sim_all = 0
        for j in range(2*n):
            for k in range(j+1, 2*n):
                sim_all += pdist(ret[j:j+1,:],ret[k:k+1,:])
        sim_all = sim_all / (n*(2*n-1))
        dist_all.update(sim_all.item())
        sim_mut = 0
        for kk in range(n):
            sim_mut += pdist(ret1[kk:kk+1,:],ret2[kk:kk+1,:])
        sim_mut = sim_mut / n
        dist_mutual.update(sim_mut.item())
    return (dist_all.avg, dist_mutual.avg)


def adjust_learning_rate(state, optimizer, epoch):
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def adjust_learning_rate_warmup(state, optimizer, epoch):
    import math
    if epoch <= 5:
        state['lr'] = args.lr * (epoch) / 5.
    else:
        if epoch in args.schedule:
            state['lr'] *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('Total time: {}hour {}min'.format(int((b - a) / 3600), int((b - a) % 3600 / 60)))
