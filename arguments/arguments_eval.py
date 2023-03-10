import argparse
import time
import os
import torch
import numpy as np
import random

parser = argparse.ArgumentParser()

'''the settings below are used for autodl'''
'''run related'''
parser.add_argument('--name', type=str,default='baseline_unet', help='name of this run')
parser.add_argument('--model', type=str, default='U_Net')
parser.add_argument('--ckpt_save_path', type=str,default='./checkpoints/')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--isTrain', default=True)
parser.add_argument('--result_save_path', type=str, default='./result/')
''' model related '''

''' dataloader  '''
parser.add_argument('--dataset_name', type=str, default='paired')
parser.add_argument('--data_root', type=str, default='./data/CRC-224/CRC-01-31-16-20')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--pin_memory', type=bool, default=True)


'''other options'''
parser.add_argument('--fixseed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=529)
parser.add_argument('--gpu_id', type=int, default=0)






def get_arguements():
    args = parser.parse_args()
    # print(args._get_kwargs())
    
    args.name = args.name + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.result_save_path = os.path.join(args.result_save_path, args.name)
    os.makedirs(args.result_save_path, exist_ok=True)
    
    arg_list = args._get_kwargs()
    with open(os.path.join(args.result_save_path, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            print("{:>20}:{:<20}".format(name, arg))
            f.write("{:>20}:{:<20}".format(name, arg)+'\n')
            
    if args.fixseed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    if args.dry_run:
        args.epoch = 1
        
    return args


if __name__ == '__main__':
    args = get_arguements()
    print(args._get_kwargs())
    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg, list):
            arg = ",".join(map(str, arg))
        print("{:>20}:{:<20}".format(name, arg))
