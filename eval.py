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

def eval_cuda(root_blurred, root_sharp):
    '''
        evaluate the sharp(blurred) image with metrics 
    '''
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


def func1(args):
    '''
        restore the image
        using ckpt and blurred images to generate sharp images
        args will be logged
    '''
    ckpt = torch.load(args.ckpt_load_path)
    
    model = get_model(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    
    eval_dataloader = get_eval_dataloader(args)
    
    t = ToPILImage()
    
    save_root = './results/{}-{}'.format(args.name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))

    os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
    
    model.eval()

    with torch.no_grad():
        for img, _, names in tqdm(eval_dataloader):
            img= img.cuda()
            preds = model(img).cpu()
            
            for pred, image_name in zip(preds, names):

                Image.fromarray(np.array(t(pred)))\
                    .save(os.path.join(save_root,'images', image_name))
            
    arg_list = args._get_kwargs()
    with open(os.path.join(save_root, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            f.write("{:>30}:{:<20}\n".format(name, arg))
    
    return save_root
        

def func2(root_blurred, root_sharp):
    '''
        evaluate the sharp(blurred) image with metrics 
    '''
    losses = []
    
    t = ToTensor()
    
    save_root = os.path.join('./results', time.strftime('%m-%d-%H-%M', time.localtime()))
    
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, 'metric values.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)
    
        for image_name in tqdm(os.listdir(root_sharp)):
            img_sharp = t(Image.open(os.path.join(root_sharp, image_name))).unsqueeze(0)
            img_blurred = t(Image.open(os.path.join(root_blurred, image_name))).unsqueeze(0)
            
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
    
def func3(root_sharp, root_blurred, root_restored, save_root = ''):
    '''
        visulize the comparison among sharp, blurred and restored image.
    '''
    ROW = 5
    COL = 4
    if save_root == '':
        save_root = os.path.join('./results', "vis-"+time.strftime('%m-%d-%H-%M', time.localtime()))
    
    
    vis_save_root = os.path.join(save_root, 'images')
    
    os.makedirs(vis_save_root, exist_ok=True)
    
    image_names = os.listdir(root_sharp)
    
    batch_size = ROW * COL
    batch_num = int(len(image_names)/batch_size) + 1
    
    for i in range(batch_num):
        plt.figure(figsize=(ROW, COL*3))
        
        for j,image_name in enumerate(image_names[i*batch_size:(i+1)*batch_size]):
            
            img_clear = np.array(Image.open(os.path.join(root_sharp, image_name)))
            img_blurred = np.array(Image.open(os.path.join(root_blurred, image_name)))
            img_restored = np.array(Image.open(os.path.join(root_restored, image_name)))
            
            plt.subplot(ROW,COL*3,3*j+1),plt.imshow(img_clear),plt.axis('off')
            plt.subplot(ROW,COL*3,3*j+2),plt.imshow(img_blurred),plt.title(image_name),plt.axis('off')
            plt.subplot(ROW,COL*3,3*j+3),plt.imshow(img_restored),plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_save_root, '{}-{}.png'.format(i*batch_size,(i+1)*batch_size)))
        plt.cla() 
        plt.clf()
        plt.close()
    
    with open(os.path.join(save_root, 'log.txt'), 'w') as f:
        f.write("Sharp root:{}\n".format(root_sharp))
        f.write("Blurred root:{}\n".format(root_blurred))
        f.write("Restored root:{}\n".format(root_restored))

def time_testing():
    time_used = np.zeros(len(metrics), dtype=np.float64)
    
    NUM_OF_IMAGES = 128
    imgs_X = torch.rand(NUM_OF_IMAGES,3,224,224)
    imgs_Y = torch.rand(NUM_OF_IMAGES,3,224,224)
    
    for x,y in tqdm(zip(imgs_X, imgs_Y)):
        for i, (_, function) in enumerate(metrics.items()):
            
            s = time.time()
            function(x.unsqueeze(0), y.unsqueeze(0))
            e = time.time()
            
            time_used[i] +=  e-s
    
    for time_u, metric in zip(time_used, metrics.keys()):
        print('{}:{:.3f}'.format(metric, time_u))
    
def direction_test():
    img_x = torch.rand(4,3,224,224)
    img_y = torch.rand(4,3,224,224)
    
    for name, function in metrics.items():
        
        print(name)
        print('x and x:{}'.format(function(img_x, img_x).data))
        print('x and y:{}'.format(function(img_x, img_y).data))
    


def test_baseline(root, type, save_root):
    # root = './data/CRC-224/'
    
    root_clear = os.path.join(root, '{}-clear'.format(type)) 
    root_blurred = os.path.join(root, '{}-blurred'.format(type))
    
    losses = []
    
    t = ToTensor()
    
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, 'metric values base.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)
    
        for image_name in tqdm(os.listdir(root_clear)):
            img_clear = t(Image.open(os.path.join(root_clear, image_name))).cuda()
            img_blurred = t(Image.open(os.path.join(root_blurred, image_name))).cuda()
            
            this_losses = []
            for _, metric_function in metrics.items():
                this_losses.append(metric_function(img_blurred.unsqueeze(0), img_clear.unsqueeze(0)).cpu().data)
            losses.append(this_losses)
            
            line = ','.join([image_name]+list(map(str, this_losses)))+'\n'
            f.write(line) 
                            
    
    losses = np.array(losses)
    # np.save(os.path.join(save_root, 'losses.npy'),losses)
    
    losses_item = np.mean(losses, axis=0)
    
    with open(os.path.join(save_root, 'result base.txt'), 'w') as f:
        for metric, value in zip(metrics.keys(), losses_item):
            f.write("{:>20}:{:<20}\n".format(metric, value))
    
    
def main():
    '''
        main:use the data to generate the 
    '''
    direction_test()
    # func2('./data/CRC-224/CRC-02-01-22-27/val-blurred', './data/CRC-224/CRC-02-01-22-27/val-clear')
    # args = get_arguements()
    
    # test_baseline(args.data_root, 'val', './results/name-2023-01-25-21-48-26')
    # all(args)

    # restore the image
    # func1(args)
    
    # func2(args.data_root)
    
    # func2()
    
    # func3()
    
        
        
def evaluate(root, save_root):
    '''
        evaluate the blurred with given sharp images
        detailed result will be saved at the root of blurred images
    '''
    root_clear = os.path.join(root, '{}-clear'.format(type)) 
    root_blurred = os.path.join(root, '{}-blurred'.format(type))
    
    losses = []
    
    t = ToTensor()
    
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, 'metric values.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)
    
        for image_name in tqdm(os.listdir(root_clear)):
            img_clear = t(Image.open(os.path.join(root_clear, image_name)))
            img_blurred = t(Image.open(os.path.join(root_blurred, image_name)))
            
            this_losses = []
            for _, metric_function in metrics.items():
                this_losses.append(metric_function(img_blurred.unsqueeze(0), img_clear.unsqueeze(0)).data)
            losses.append(this_losses)
            
            line = ','.join([image_name]+list(map(str, this_losses)))+'\n'
            f.write(line) 
                            
    
    losses = np.array(losses)
    np.save(os.path.join(save_root, 'losses.npy'),losses)
    
    losses_item = np.mean(losses, axis=0)
    
    with open(os.path.join(save_root, 'result.txt'), 'w') as f:
        for metric, value in zip(metrics.keys(), losses_item):
            f.write("{:>20}:{:<20}\n".format(metric, value))
    ...        
        

def all(args):
    ckpt = torch.load(args.ckpt_load_path)
    
    model = get_model(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    
    eval_dataloader = get_eval_dataloader(args)
    
    losses = []
    
    t = ToPILImage()
    
    save_root = './results/{}-{}'.format(args.name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
    
    model.eval()
    with open(os.path.join(save_root, 'metric values.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)

        with torch.no_grad():
            for img, labels, names in tqdm(eval_dataloader):
                img = img.cuda()
                labels = labels.cuda()
                preds = model(img)
                # preds = model(img).cpu()
                
                for pred, label, image_name in zip(preds, labels, names):
                    this_losses = []
                    for _, metric_function in metrics.items():
                        this_losses.append(
                            metric_function(
                                pred.unsqueeze(0), 
                                label.unsqueeze(0)).cpu().data)
                        
                    losses.append(this_losses)

                    line = ','.join([image_name]+list(map(str, this_losses)))+'\n'
                    f.write(line) 
                    
                    Image.fromarray(np.array(t(pred)))\
                        .save(os.path.join(save_root,'images', image_name))
                        
    losses = np.array(losses)

    np.save(os.path.join(save_root, 'losses.npy'),losses)
    losses = np.mean(losses, axis=0)
    
    with open(os.path.join(save_root, 'result.txt'), 'w') as f:
        for metric, value in zip(metrics.keys(), losses):
            f.write("{:>20}:{:<20}\n".format(metric, value))
            
    arg_list = args._get_kwargs()
    with open(os.path.join(save_root, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            f.write("{:>30}:{:<20}\n".format(name, arg))
        
    batch_vis(args.data_root, save_root)        
        
        
def restore_image_and_eval(args):
    ckpt = torch.load(args.ckpt_load_path)
    
    model = get_model(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    
    eval_dataloader = get_eval_dataloader(args)
    
    losses = []
    
    t = ToPILImage()
    
    save_root = './results/{}-{}'.format(args.name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
    
    model.eval()
    with open(os.path.join(save_root, 'metric values.csv'), 'w') as f:
        line = ','.join(['image name']+list(metrics.keys()))+'\n'
        f.write(line)

        with torch.no_grad():
            for img, labels, names in tqdm(eval_dataloader):
                img= img.cuda()
                
                preds = model(img).cpu()
                # preds = torch.rand(img.shape)
                
                for pred, label, image_name in zip(preds, labels, names):
                    this_losses = []
                    for _, metric_function in metrics.items():
                        this_losses.append(metric_function(pred.unsqueeze(0), label.unsqueeze(0)).data)
                    losses.append(this_losses)

                    line = ','.join([image_name]+list(map(str, this_losses)))+'\n'
                    f.write(line) 
                    
                    if args.visulize:
                        Image.fromarray(np.array(t(pred)))\
                            .save(os.path.join(save_root,'images', image_name))
                        
    losses = np.array(losses)

    np.save(os.path.join(save_root, 'losses.npy'),losses)
    losses = np.mean(losses, axis=0)
    
    with open(os.path.join(save_root, 'result.txt'), 'w') as f:
        for metric, value in zip(metrics.keys(), losses):
            f.write("{:>20}:{:<20}\n".format(metric, value))
            
    arg_list = args._get_kwargs()
    with open(os.path.join(save_root, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            f.write("{:>30}:{:<20}\n".format(name, arg))
        
    if args.visulize:
        batch_vis(args.data_root, save_root)
    
def restore_image(args): 
    '''
        load model,
        restore images,
        log the args.
    '''       
    ckpt = torch.load(args.ckpt_load_path)
    
    model = get_model(args)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    
    eval_dataloader = get_eval_dataloader(args)
    
    t = ToPILImage()
    
    save_root = './results/{}-{}'.format(args.name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for img, names in tqdm(eval_dataloader):
            img= img.cuda()
            
            preds = model(img).cpu()
            
            for pred, image_name in zip(preds, names):
                Image.fromarray(np.array(t(pred)))\
                    .save(os.path.join(save_root,'images', image_name))
            
    arg_list = args._get_kwargs()
    with open(os.path.join(save_root, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            f.write("{:>30}:{:<20}\n".format(name, arg))
        
        
def batch_vis(data_root, save_root):
    ROW = 6
    COL = 4
    
    clear_image_root = os.path.join(data_root, 'val-clear')
    blurred_image_root = os.path.join(data_root, 'val-blurred')
    restored_image_root = os.path.join(save_root, 'images')

    vis_save_root = os.path.join(save_root, 'vis')
    os.makedirs(vis_save_root, exist_ok=True)
    
    image_names = os.listdir(clear_image_root)
    
    batch_size = ROW * COL
    batch_num = int(len(image_names)/batch_size) + 1
    
    for i in tqdm(range(batch_num)):
        plt.figure(figsize=(COL*3*2, ROW*2))
        
        for j,image_name in enumerate(image_names[i*batch_size:(i+1)*batch_size]):
            
            img_clear = np.array(Image.open(os.path.join(clear_image_root, image_name)))
            img_blurred = np.array(Image.open(os.path.join(blurred_image_root, image_name)))
            img_restored = np.array(Image.open(os.path.join(restored_image_root, image_name)))
            
            plt.subplot(ROW,COL*3,3*j+1),plt.imshow(img_blurred),plt.axis('off')
            plt.subplot(ROW,COL*3,3*j+2),plt.imshow(img_clear),plt.title(image_name),plt.axis('off')
            plt.subplot(ROW,COL*3,3*j+3),plt.imshow(img_restored),plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_save_root, '{}-{}.png'.format(i*batch_size,(i+1)*batch_size)))
        # plt.show(block=True)
        # return 
        plt.cla() 
        plt.clf()
        plt.close()
        
    
if __name__ == '__main__':
    # time_testing()
    # args = get_arguements()
    # save_root = './results/name-2023-01-22-21-59-15'
    # batch_vis(args, save_root)
    ...
    main()
    
    
    # x = torch.rand(3,256,256)
    # print(x.shape)
    # x = x.unsqueeze(0)
    # print(x.shape)
    # line = ','.join(['image name']+list(metrics.keys()))
    # line = ','.join(['image name']+list(map(str,[1.21,2.43,3.32,4.32])))
    # print(line)
    # x=torch.rand(4,3,256,256)
    # y=torch.rand(4,3,256,256)
    
    # # print(torch.max(x), torch.min(x))
    # # result = eval_metrics(x,y)
    
    
    # # print(eval_metrics(x,x))
    # print(eval_metrics(x,y))
    
    # arr = [
    #     [1,2,3,4],
    #     [3,4,6,7],
    #     [5,6,9,1],
    # ]
    # arr = np.array(arr)
    
    # # for row in arr:
    # #     for item in row:
    # #         print(item)
    
    # print(np.mean(arr, axis = 0))

