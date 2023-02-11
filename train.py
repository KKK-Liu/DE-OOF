import torch
from models.__init__ import get_model
from arguments.arguments_train import get_arguements
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


import os


from dataloader.dataloader_paired import get_dataloader
from tqdm import tqdm

from utils.utils import AverageMeter
from utils.loss_function import get_loss_function
from utils.Log import Logger
from torch.utils.tensorboard.writer import SummaryWriter

import time

def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    
    '''
        Initialization!
    '''

    args = get_arguements()

    logger = Logger(args)

    train_dataloader, val_dataloader= get_dataloader(args)    
    model = get_model(args).cuda()
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        nesterov=True, 
        weight_decay=args.weight_decay)
    
    scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)

    loss_function = get_loss_function(args).cuda()

    start_epoch = 1
    best_val_loss = None
    
    train_loss = torch.tensor(0.0).cuda()
    train_loss.requires_grad_(True)
    val_loss = torch.tensor(0.0).cuda()

    train_loss_recoder = AverageMeter()
    val_loss_recoder = AverageMeter()

    args.ckpt_save_path = os.path.join(args.ckpt_save_path,args.name) 

    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)

        
    if args.use_tensorboard:
        tw = SummaryWriter('/root/tf-logs/runs/'+args.name+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+'/'+args.model+'/')
    
        
    progress_bar = tqdm(range(start_epoch, args.epoch+1), desc='Epoch')
    msg = 'train loss: {:.6f} val loss: {:.6f}  best val loss: {:.6f} '.format(
        train_loss_recoder.avg, val_loss_recoder.avg, best_val_loss
    )
    progress_bar.set_postfix_str(msg)

    '''
        Fitting!
    '''
    for epoch in progress_bar:
        train_loss_recoder.reset()
        val_loss_recoder.reset()

        '''
            Train!
        '''
        model.train()
        for img, label in tqdm(train_dataloader):
            img, label = img.cuda(), label.cuda()
            
            prediction = model(img)

            train_loss = loss_function(prediction, label)

            train_loss_recoder.update(train_loss.item(), n=label.size(0))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # print(1)

        '''
            Validation!
        '''
        model.eval()
        with torch.no_grad():
            for img, label in tqdm(val_dataloader):
                img, label = img.cuda(), label.cuda()
                prediction = model(img)

                val_loss = loss_function(prediction, label)

                val_loss_recoder.update(val_loss.item(), n=label.size(0))

        '''
            Log!
        '''
        msg = 'train loss: {:.6f} val loss: {:.6f}  best val loss: {:.6f} '.format(
            train_loss_recoder.avg, val_loss_recoder.avg, best_val_loss
        )
        progress_bar.set_postfix_str(msg)
        logging.info('Epoch: {:0>3} '.format(epoch)+msg)
        
        
        if best_val_loss == None or val_loss_recoder.avg < best_val_loss:
            best_val_loss = val_loss_recoder.avg

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }

            torch.save(state,os.path.join(args.ckpt_save_path, '{}_{}_valBest_{:.5f}_ckpt.pth.tar'.format(args.model, best_val_loss)))
            
        if args.use_tensorboard and epoch % args.tensorboard_freq == 0:
            tw.add_scalar('train loss',             train_loss_recoder.avg,epoch)
            tw.add_scalar('validation loss',        val_loss_recoder.avg,epoch)
            tw.add_scalar('best val loss',          best_val_loss,epoch)

        scheduler.step()
        

if __name__ == '__main__':
    main()
