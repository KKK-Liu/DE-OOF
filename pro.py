import torch
from models import get_model
from arguments.arguments_train import get_arguements

import os
import wandb
from dataloader import get_dataloader
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import time

def main():
    '''
        Initialization!
    '''
    args = get_arguements()
    train_dataloader, val_dataloader= get_dataloader(args)    
    model = get_model(args)
    model.to_cuda()

    if args.use_tensorboard:
        tw = SummaryWriter(os.path.join('/root/tf-logs/runs', args.name))
        
    if args.use_wandb:
        wandb.init(
            entity = 'kkk06',
            project = 'Deblur',
            name = args.name,
            config = vars(args)
        )
        
        wandb.watch(
            models = model.net,
            log='all',
            log_freq=32
        )

    # progress_bar = tqdm(range(1, args.epoch+1), desc='Epoch')
    # progress_bar.set_postfix_str(model.get_log_message()+' train time:{:.0f}, val time:{:.0f}'.format(0,0))

    '''
        Fitting!
    '''
    s = time.time()
    for epoch in range(1, args.epoch+1):
        model.epoch_start()
        '''
            Train!
        '''
        model.mode('train')
        train_s = time.time()
        for data in train_dataloader:
            model.set_input(data)
            model.train_step()
            
            if args.dry_run:
                break
        train_e = time.time()

        '''
            Validation!
        '''
        model.mode('valid')
        val_s = time.time()
        for data in val_dataloader:            
            model.set_input(data)
            model.valid_step()
            
            if args.dry_run:
                break
        val_e = time.time()

        model.epoch_finish(epoch)
        
        '''
            Log!
        '''
        n = time.time()
        msg = 'Epoch:{:0>3d} {} train time:{:.0f}, val time:{:.0f} Finish at {}'.format(
            epoch, model.get_log_message(),train_e-train_s,val_e-val_s, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(
                (n-s)/epoch * (args.epoch-epoch) +n 
            )))
        print(msg)
        # progress_bar.set_postfix_str(msg)
        
        if args.use_tensorboard:
            scalar_dict = model.get_scalar_dict()
            for k,v in scalar_dict.items():
                tw.add_scalar(k,v,epoch)
             
        if args.use_wandb:
            scalar_dict = model.get_scalar_dict()
            wandb.log(scalar_dict)



if __name__ == '__main__':
    main()
