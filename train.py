import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
# custom module
from util import *
from src.dataset import *
from src.model import EfficientDet

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
device = torch.device("cuda")


def get_args():
    # Get Arguments
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--batch_size", type=int,
                        default=450)
    parser.add_argument("--lr", type=float,
                        default=1e-4)
    parser.add_argument("--num_epochs", type=int,
                        default=200)
    parser.add_argument("--expname", type=str,
                        default='EXP', help="experiment name")
    parser.add_argument("--resume", type=str,
                        default="/checkpoint.pth.tar",
                        help="resume checkpoint")
    parser.add_argument("--best", type=str,
                        default="/HW2model.pth.tar", help="best model")
    parser.add_argument("--io", type=str,
                        default="/run.log")

    args = parser.parse_args()
    args.best_loss = 1e5
    args.start_epoch = 0

    if not os.path.exists(args.expname):
        os.mkdir(args.expname)
    args.resume = args.expname + args.resume
    args.best = args.expname + args.best
    args.io = IOStream(args.expname + args.io)

    return args


def LoadData():
    ''' Randomly split data into 33000 training data
        and 402 data for validation
    Set seed to make sure that I can get the same
    training data at all time '''
    np.random.seed(0)
    perm = np.random.permutation(33402)
    TrainIndices = perm[:33000]
    TestIndices = perm[33000:]

    training_set = TrainDataset(TrainIndices)
    test_set = ValDataset(TestIndices)

    training_params = {"batch_size": args.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 32}
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": 32}

    train_loader = DataLoader(training_set, **training_params)
    test_loader = DataLoader(test_set, **test_params)

    return train_loader, test_loader


def main(args):
    ''' main function '''
    torch.cuda.manual_seed(1)

    # load data
    train_loader, test_loader = LoadData()

    # define model
    model = EfficientDet(num_classes=10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # resume previous checkppoint if possible
    if os.path.isfile(args.resume):
        print('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded checkpoint {}(epoch {})'
              .format(args.resume, args.start_epoch))
        print('best loss:', args.best_loss)
    else:
        print('no checkpoint found at {}'.format(args.resume))

    # define optimizer and schduler
    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                  weight_decay=1e-5, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1,
                                  patience=3, verbose=True)

    # start training
    for epoch in range(args.start_epoch, args.num_epochs):
        args.io.cprint('Epoch[%d]' % epoch)

        model.train()

        # train
        train(model, train_loader, optimizer, args)

        # validation
        model.eval()
        loss = val(model, test_loader, args)

        # save checkpoint
        is_best = loss < args.best_loss
        args.best_loss = min(loss, args.best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': args.best_loss,
        }, is_best, args)

        scheduler.step(loss)


def train(model, train_loader, optimizer, args):
    ''' Training '''

    # Object for visualizing
    clsLoss = AverageMeter()
    regLoss = AverageMeter()
    Loss = AverageMeter()

    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()

        # get image and annotation
        image = sample['img']
        annotation = sample['annot']
        batch_size = image.shape[0]

        image = image.to(device).float()
        annotation = annotation.to(device)

        # get losses
        cls_loss, reg_loss = model([image, annotation])

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss

        # back propagation
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        # visualize losses
        clsLoss.update(cls_loss, batch_size)
        regLoss.update(reg_loss, batch_size)
        Loss.update(loss, batch_size)

        print('[{0}/{1}]  '
              'Loss {Loss.val:.4f} ({Loss.avg:.4f})  '
              'clsLoss {clsLoss.val:.4f} ({clsLoss.avg:.4f})  '
              'regLoss {regLoss.val:.4f} ({regLoss.avg:.4f})         '
              .format(i + 1, len(train_loader),
                      Loss=Loss,
                      clsLoss=clsLoss,
                      regLoss=regLoss), end='\r')

    # save log
    print(' ' * 100, end='\r')
    args.io.cprint(' * Train * Loss {Loss.avg:.4f}  '
                   'clsLoss {clsLoss.avg:.4f}  '
                   'regLoss {regLoss.avg:.4f}'
                   .format(Loss=Loss,
                           clsLoss=clsLoss,
                           regLoss=regLoss))


def val(model, test_loader, args):
    ''' Validation '''

    # Object for visualizing
    clsLoss = AverageMeter()
    regLoss = AverageMeter()
    Loss = AverageMeter()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            # get image and annotation
            image = sample['img']
            annotation = sample['annot']
            batch_size = image.shape[0]

            image = image.to(device).float()
            annotation = annotation.to(device)

            # get losses
            cls_loss, reg_loss = model([image, annotation])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()

            # visualize losses
            clsLoss.update(cls_loss, batch_size)
            regLoss.update(reg_loss, batch_size)
            Loss.update(cls_loss + reg_loss, batch_size)

            print('[{0}/{1}]  '
                  'Loss {Loss.val:.4f} ({Loss.avg:.4f})  '
                  'clsLoss {clsLoss.val:.4f} ({clsLoss.avg:.4f})  '
                  'regLoss {regLoss.val:.4f} ({regLoss.avg:.4f})         '
                  .format(i + 1, len(test_loader),
                          Loss=Loss,
                          clsLoss=clsLoss,
                          regLoss=regLoss), end='\r')

    # Save log
    print(' ' * 100, end='\r')
    args.io.cprint(' * Test * Loss {Loss.avg:.4f}  '
                   'clsLoss {clsLoss.avg:.4f}  '
                   'regLoss {regLoss.avg:.4f}'
                   .format(Loss=Loss,
                           clsLoss=clsLoss,
                           regLoss=regLoss))
    return Loss.avg


if __name__ == "__main__":
    args = get_args()
    main(args)
