from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

import torch
from arguments.arguments_eval import get_arguements
from models import get_model
from dataloader.dataloader_paired import get_eval_dataloader
import time


from piq import ssim, multi_scale_ssim,multi_scale_gmsd, vif_p, StyleLoss, ContentLoss, LPIPS, DISTS, psnr, fsim, vsi, mdsi, haarpsi, srsim, PieAPP, dss, information_weighted_ssim

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

''' run '''
parser.add_argument('--name',type=str, default='name')

''' dataloader '''

''' model '''

parser.add_argument('--ckpt_load_path', type=str,default='')
parser.add_argument('--choice',type=int, default=0,help='0-eval,1-inferrence and save, 2-inferrence and eval, 3-inferrence and eval and save')
parser.add_argument('--data_root',type=str, default='')
parser.add_argument('--save_root', type=str, default='')
parser.add_argument('--sharp_root')

metrics = {
    'mse':torch.nn.functional.mse_loss,
    'psnr':psnr,
    'ssim':ssim,
    'multi_scale_ssim':multi_scale_ssim,
    'multi_scale_gmsd':multi_scale_gmsd,
    'vif_p':vif_p,
    'mdsi':mdsi,
    'haarpsi':haarpsi,
    'srsim':srsim,
    'dss':dss,
    'information_weighted_ssim':information_weighted_ssim,
    'fsim':fsim,
    'vsi':vsi,
    
    # metrics below are time-consuming
    # 'StyleLoss':StyleLoss(),
    # 'ContentLoss':ContentLoss(),
    # 'LPIPS':LPIPS(),
    # 'DISTS':DISTS(),
    # 'PieAPP':PieAPP(),
}

# whether to inferrence
# whether to save
# whether to eval


def eval(root_sharp:str, root_blurred:str):
    losses = []
    
    t = ToTensor()
    
    save_root = os.path.join('./results', time.strftime('%m-%d-%H-%M', time.localtime()))
    
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, 'metric values.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)
        for image_name in tqdm(os.listdir(root_sharp)):
            img_sharp = t(Image.open(os.path.join(root_sharp, image_name))).unsqueeze(0).cuda()
            img_blurred = t(Image.open(os.path.join(root_blurred, image_name))).unsqueeze(0).cuda()
            
            this_losses = []
            for _, metric_function in metrics.items():
                this_losses.append(metric_function(img_blurred, img_sharp).data)
            losses.append(this_losses)
            
            line = ','.join([image_name]+list(map(str, this_losses)))+'\n'
            f.write(line)

    losses = np.array(losses)
    np.save(os.path.join(save_root, 'losses.npy'),losses)
    
    losses_item = np.mean(losses, axis=0)
    
    with open(os.path.join(save_root, 'result.txt'), 'w') as f:
        f.write("blur root:{}\n".format(root_blurred))
        f.write("sharp root:{}\n".format(root_sharp))
        for metric, value in zip(metrics.keys(), losses_item):
            f.write("{:>20}:{:<20}\n".format(metric, value))
    
    print('Results are saved in {}'.format(os.path.join(save_root, 'result.txt')))
    
    with open(os.path.join(save_root, 'result.txt'), 'r') as f:
        for line in f.readlines():
            print(line)
            
    return save_root

def inferrence_save_and_eval(args):
    
    model = get_model(args)
    model.to_cuda()
    
    dataloader = get_eval_dataloader(args)
    
    model.mode('valid')
    for data in dataloader:
        model.set_input(data)
        model.get_visuals()
        
        if args.choice in [2,3]:
            model.eval_visuals(metrics)
            
        if args.choice in [1,3]:
            model.save_visuals()
            
    if args.choice in [2,3]:
        model.eval_result_save(metrics)
            
            
        
    
def main():
    args = parser.parse_args()
    ''' 
        0-eval,
        1-inferrence and save,
        2-inferrence and eval,
        3-inferrence and eval and save 
    '''
    if args.choice == 0:
        eval(args.root_sharp, args.root_blurred)
    elif args.choice in [1,2,3]:
        inferrence_save_and_eval(args)
    else:
        raise NotImplementedError("{} is not supported".format(args.choice))
    ...
if __name__ == '__main__':
    main()
    ...