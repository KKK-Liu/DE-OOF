import torch
from models import get_model
from arguments.arguments_train import get_arguements

import os

from dataloader import get_dataloader
from tqdm import tqdm
from utils.Log import Logger
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    '''
        Initialization!
    '''
    args = get_arguements()
    # logger = Logger(args)
    # The existance of tensorboard makes logger useless.
    train_dataloader, val_dataloader= get_dataloader(args)    
    model = get_model(args)
    model.to_cuda()

    if args.use_tensorboard:
        tw = SummaryWriter(os.path.join('/root/tf-logs/runs', args.name))
      
    progress_bar = tqdm(range(1, args.epoch+1), desc='Epoch')
    progress_bar.set_postfix_str(model.get_log_message())

    '''
        Fitting!
    '''
    for epoch in progress_bar:
        model.epoch_start()
        '''
            Train!
        '''
        model.mode('train')
        for data in tqdm(train_dataloader):
            model.set_input(data)
            model.train_step()
            
            if args.dry_run:
                break

        '''
            Validation!
        '''
        model.mode('valid')
        for data in tqdm(val_dataloader):            
            model.set_input(data)
            model.valid_step()
            
            if args.dry_run:
                break

        model.epoch_finish(epoch)
        
        '''
            Log!
        '''

        msg = model.get_log_message()
        
        progress_bar.set_postfix_str(msg)
        # logger('Epoch: {:0>3} '.format(epoch)+msg)
        
        if args.use_tensorboard:
            scalar_dict = model.get_scalar_dict()
            for k,v in scalar_dict.items():
                tw.add_scalar(k,v,epoch)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    main()
