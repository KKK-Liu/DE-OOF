import torch
from models import get_model
from arguments.arguments_eval import get_arguements

import os
import wandb
from dataloader.dataloader_paired_3 import get_eval_dataloader
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import time

def main():
    '''
        Initialization!
    '''
    args = get_arguements()
    val_dataloader= get_eval_dataloader(args)    
    model = get_model(args)
    model.load_network()
    model.to_cuda()

    s = time.time()

    model.mode('valid')
    for data in val_dataloader:
        model.set_input(data)
        model.get_and_save_visual_results()




if __name__ == '__main__':
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    main()
