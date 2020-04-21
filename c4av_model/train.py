import os
import argparse
import json
import shutil
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import Talk2Car
from utils.collate import custom_collate
from utils.util import AverageMeter, ProgressMeter, save_checkpoint

import models.resnet as resnet
import models.nlp_models as nlp_models

parser = argparse.ArgumentParser(description='Talk2Car object referral')
parser.add_argument('--root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=18, type=int,
                    metavar='N',
                    help='mini-batch size (default: 18)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--milestones', default=[4, 8], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='nesterov')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

def main():
    args = parser.parse_args()

    # Create dataset
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    train_dataset = Talk2Car(root=args.root, split='train',
                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
    val_dataset = Talk2Car(root=args.root, split='val',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,                            drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,                            drop_last=False)

    # Create model
    print("=> creating model")
    img_encoder = resnet.__dict__['resnet18'](pretrained=True) 
    text_encoder = nlp_models.TextEncoder(input_dim=train_dataset.number_of_words(),
                                                 hidden_size=512, dropout=0.1)
    img_encoder.cuda()
    text_encoder.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index = train_dataset.ignore_index, 
                                    reduction = 'mean')
    criterion.cuda()    
    
    cudnn.benchmark = True

    # Optimizer and scheduler
    print("=> creating optimizer and scheduler")
    params = list(text_encoder.parameters()) + list(img_encoder.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                            gamma=0.1)

    # Checkpoint
    checkpoint = 'checkpoint.pth.tar'
    if os.path.exists(checkpoint):
        print("=> resume from checkpoint at %s" %(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_ap50 = checkpoint['best_ap50']
    else:
        print("=> no checkpoint at %s" %(checkpoint))
        best_ap50 = 0
        start_epoch = 0

    # Start training
    print("=> start training")

    for epoch in range(start_epoch, args.epochs):
        print('Start epoch %d/%d' %(epoch, args.epochs))
        print(20*'-')

        # Train 
        train(train_dataloader, img_encoder, text_encoder, optimizer, criterion, epoch, args)
        
        # Update lr rate
        scheduler.step()
        
        # Evaluate
        ap50 = evaluate(val_dataloader, img_encoder, text_encoder, args)
        
        # Checkpoint
        if ap50 > best_ap50:
            new_best = True
            best_ap50 = ap50
        else:
            new_best = False

        save_checkpoint({'img_encoder': img_encoder.state_dict(),
                         'text_encoder': text_encoder.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'epoch': epoch + 1, 'best_ap50': best_ap50}, new_best = new_best)

    # Evaluate
    if args.evaluate:
        print("=> Evaluating best model")
        checkpoint = torch.load('best_model.pth.tar', map_location='cpu')
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        ap50 = evaluate(val_dataloader, img_encoder, text_encoder, args)
        print('AP50 on validation set is %.2f' %(ap50*100))

def train(train_dataloader, img_encoder, text_encoder, optimizer, criterion,
            epoch, args):
    m_losses = AverageMeter('Loss', ':.4e')
    m_top1 = AverageMeter('Acc@1', ':6.2f')
    m_iou = AverageMeter('IoU', ':6.2f')
    m_ap50 = AverageMeter('AP50', ':6.2f')
    progress = ProgressMeter(
                len(train_dataloader),
                [m_losses, m_top1, m_iou, m_ap50], prefix="Epoch: [{}]".format(epoch))
 
    img_encoder.train()
    text_encoder.train()
    
    ignore_index = train_dataloader.dataset.ignore_index
        
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)
        gt = batch['rpn_gt'].cuda(non_blocking=True)
        iou = batch['rpn_iou'].cuda(non_blocking=True)
        b, r, c, h, w = region_proposals.size()

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm)
       
        # Sentence features
        _, sentence_features = text_encoder(command.permute(1,0), command_length)
        norm = sentence_features.norm(p=2, dim=1, keepdim=True)
        sentence_features = sentence_features.div(norm)
     
        # Product in latent space
        scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()

        # Loss
        total_loss = criterion(scores, gt)

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Summary
        pred = torch.argmax(scores, 1)
        pred_bin = F.one_hot(pred, r).bool()
        valid = (gt!=ignore_index)
        num_valid = torch.sum(valid).float().item()
        m_top1.update(torch.sum(pred[valid]==gt[valid]).float().item(), num_valid)
        m_iou.update(torch.masked_select(iou, pred_bin).sum().float().item(), b)
        m_ap50.update((torch.masked_select(iou, pred_bin) > 0.5).sum().float().item(), b)
        m_losses.update(total_loss.item())

        if i % args.print_freq==0:
            progress.display(i)
    

@torch.no_grad()
def evaluate(val_dataloader, img_encoder, text_encoder, args):
    m_top1 = AverageMeter('Acc@1', ':6.2f')
    m_iou = AverageMeter('IoU', ':6.2f')
    m_ap50 = AverageMeter('AP50', ':6.2f')
    progress = ProgressMeter(
                len(val_dataloader),
                [m_top1, m_iou, m_ap50], 
                prefix='Test: ')
 
    img_encoder.eval()
    text_encoder.eval()
    
    ignore_index = val_dataloader.dataset.ignore_index
 
    for i, batch in enumerate(val_dataloader):
        
        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)
        b, r, c, h, w = region_proposals.size()

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm)
       
        # Sentence features
        _, sentence_features = text_encoder(command.permute(1,0), command_length)
        norm = sentence_features.norm(p=2, dim=1, keepdim=True)
        sentence_features = sentence_features.div(norm)
     
        # Product in latent space
        scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()

        # Summary
        pred = torch.argmax(scores, 1)
        gt = batch['rpn_gt'].cuda(non_blocking=True)
        iou = batch['rpn_iou'].cuda(non_blocking=True)
        pred_bin = F.one_hot(pred, r).bool()
        valid = (gt!=ignore_index)
        num_valid = torch.sum(valid).float().item()
        m_top1.update(torch.sum(pred[valid]==gt[valid]).float().item(), num_valid)
        m_iou.update(torch.masked_select(iou, pred_bin).sum().float().item(), b)
        m_ap50.update((torch.masked_select(iou, pred_bin) > 0.5).sum().float().item(), b)
        if i % args.print_freq==0:
            progress.display(i)

    return m_ap50.avg   

main()
