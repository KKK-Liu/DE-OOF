import argparse

parser = argparse.ArgumentParser()

'''the settings below are used for autodl'''
''' model related '''

parser.add_argument('--gpu_ids', type=int, default=0)

''' dataloader related '''
parser.add_argument('--data_root', type=str, default='./data/CRC-224/CRC-01-25-13-12')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)


'''run related'''
parser.add_argument('--model', type=str, default='U_Net')
parser.add_argument('--ckpt_load_path', type=str,default='./checkpoints/baseline_unet2023-01-25-13-21-45baseline_unet_U_Net_valBest_0.0017_ckpt.pth.tar')
parser.add_argument('-name', type=str, default='name')


'''other options'''
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--visulize', action='store_false')

parser.add_argument('--choice', type=int, default=1, help='''
1. use ckpt and blurred image to generate sharp images
2. use ckpt and blurred image to generate sharp images and evaluate at the same time 
3. evaluate images
4. use sharp image, blurred image and restored image to form comparison data
5. generate, evaluate, compare.
''')



args = parser.parse_args()


def get_arguements():
    # print(args._get_kwargs())
    import os
    arg_list = args._get_kwargs()
    os.makedirs('./logs', exist_ok=True)
    with open('./logs/{}.txt'.format(args.name), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            print("{:>20}:{:<20}".format(name, arg))
            f.write("{:>20}:{:<20}".format(name, arg)+'\n')
        
    return args


if __name__ == '__main__':
    args = get_arguements()
    print(args._get_kwargs())
    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg, list):
            arg = ",".join(map(str, arg))
        print("{:>20}:{:<20}".format(name, arg))
