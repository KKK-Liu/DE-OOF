import matplotlib.pyplot as plt
import os
from itertools import product
from PIL import Image
import numpy as np
import cv2
def comparison_graph():
    root = ''
    image_names = []

def psf_show():
    root = './plots_npy/color-real-3um'
    save_root = './results/figs'
    os.makedirs(save_root, exist_ok=True)
    for name in ['R', 'G', 'B']:
        plt.figure(figsize=(8*4,4*4))
        for i in range(31):
            file_name = '{}-{:0>2}.npy'.format(name, i)
            psf = np.load(os.path.join(root, file_name))
            
            plt.subplot(4,8,i+1),plt.imshow(psf),plt.axis('off'),plt.title('{}-{:0>2}'.format(name, i))
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, "3um-raw_{}.png".format(name)))
        # plt.show()
        # return 
            
    for name in ['R', 'G', 'B']:
        plt.figure(figsize=(8*4,4*4))
        for i in range(32):
            file_name = '{}-{:0>2}.npy'.format(name, i)
            psf = np.load(os.path.join(root, file_name))
            psf = cv2.resize(psf, (21,21))
            plt.subplot(4,8,i+1),plt.imshow(psf),plt.axis('off'),plt.title('{}-{:0>2}'.format(name, i))
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, "3um-DownSampled_21_{}.png".format(name)))
    
def vis_batch():
    sharp_image_root = ''
    image_names = os.listdir(sharp_image_root)
    image_roots = [
        ('Blurred',''),
        ('Ground Truth',''),
        ('CycleGAN',''),
        ('DeepDeblur',''),
        ('SRN-DeblurNet',''),
        ('DeblurGAN-v2',''),
        ('Ours',''),
    ]
    
    
    BATCH_SIZE = 8
    
    row = BATCH_SIZE
    col = len(image_roots)
    
    image_batches = [image_names[i:i+BATCH_SIZE] for i in range(0,len(image_names), BATCH_SIZE)]
    for image_batch in image_batches:
        plt.figure(figsize=(len(image_roots)*2, BATCH_SIZE*2))
        for pos,(i,j) in enumerate(product(range(row), range(col))):
            plt.subplot(i, j, pos)
            image_root = os.path.join(image_roots[j][1], image_batch[i])
            img = Image.open(image_root)
            plt.imshow(img)
            
    
def get_height_map(vmin = 0, vmax = 63):
    import numpy as np
    kernel_mapping = np.random.randint(low = 0, high = 70, size=(224,224))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=8, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed) + vmin
    kernel_mapping_smoothed = kernel_mapping_smoothed * 3
    tmp_m = vmax - np.max(kernel_mapping_smoothed)
    
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    return kernel_mapping_smoothed
    
def good_vis_kernel(vmin=0, vmax=16):
    import matplotlib.pyplot as plt
    import matplotlib

    # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(nrows=4, ncols=6)
    for ax in axes.flat:
        
        height_map = get_height_map()
        
        im = ax.imshow(height_map , vmin = vmin, vmax = vmax )
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()


def generate_blur_mask_raw():
    src_root = r"D:\desktop\de-OOF\data\CRC-224\CRC-02-16-17-08\val-clear"
    save_root = './results/vis'
    os.makedirs(save_root, exist_ok=True)
    
    for index, image_root in enumerate(os.listdir(src_root)):
        image_name = image_root.replace('.png','')
        
        ori_img = Image.open(os.path.join(src_root, image_root))
        
        ori_img.save(os.path.join(save_root,'{}_ori_{}.png'.format(image_name, index)))
        
        img = np.array(ori_img)
        
        kernel_mapping = np.random.randint(low = 0, high = 70, size=(224,224))
        kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
        
        kernel_mapping_smoothed = kernel_mapping.copy()
        kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
        kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=8, ksize=(0,0))
        
        kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
        
        kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
        
        tmp_m = 60 - np.max(kernel_mapping_smoothed)
        
        for i in range(tmp_m):
            kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
        kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
        
        kernel_mapping_smoothed = np.dstack([kernel_mapping_smoothed, kernel_mapping_smoothed, kernel_mapping_smoothed])
        tmp_kernel = np.array(kernel_mapping_smoothed, dtype=np.float32)
        tmp_kernel = tmp_kernel - np.min(tmp_kernel)
        tmp_kernel = tmp_kernel/np.max(tmp_kernel)*255
        tmp_kernel = np.array(tmp_kernel, dtype=np.uint8)
        Image.fromarray(tmp_kernel).save(os.path.join(save_root,'{}_blurmask_{}.png'.format(image_name, index)))
        # exit()
        
        Image.fromarray(255 - tmp_kernel).save(os.path.join(save_root,'{}_sharpmask_{}.png'.format(image_name, index)))
        # exit()
        
        result_image = np.zeros(img.shape, dtype=np.uint8)
        
        for i, name in zip(range(3), ['B', 'G', 'R']):
            z_min = np.min(kernel_mapping_smoothed[:,:,i])
            z_max = np.max(kernel_mapping_smoothed[:,:,i])
            
            for j in range(z_min, z_max + 1):
                kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
                
                kernel = kernel/np.sum(kernel)
                kernel = cv2.Mat(kernel*255)
                kernel = cv2.resize(kernel,(21,21))
                kernel = kernel/np.sum(kernel)
                
                blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
                blurred_one_channel = np.array(blurred_one_channel)
                
                result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
        result_image = np.array(result_image, dtype=np.uint8)
        
        Image.fromarray(result_image).save(os.path.join(save_root,'{}_blurred_{}.png'.format(image_name, index)))
    
    
def vis_kernel_height_blurred():
    '''
        4*4 images:
            2*4 kernel(0~32)
            2*2 height
            2*2 blurred(according to the height)
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(nrows = 4, ncols = 4)
    axes:np.ndarray
    # axes[0][0].plot([i for i in range(1000)],[i**2 for i in range(1000)])
    kernels = []
    
    for axis in axes.flatten():
        # axis.set_axes('off')
        axis.tick_params(bottom=False,top=False,left=False,right=False)
        # axis.spine_params(bottom=False,top=False,left=False,right=False)


    
    # for i,j in product(range(2), repeat=2):
    #     k = get_kernel()
    #     kernels.append(k)
    #     axes[i][j].imshow(k, cmap='gray')
    plt.show()
    
    
def blur_one_image_RGB_known_height(ori_img, height_map):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)
    
    height_map = np.dstack([height_map, height_map, height_map])
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(height_map[:,:,i])
        z_max = np.max(height_map[:,:,i])
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(21,21))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[height_map[:,:,i]==j, i] = blurred_one_channel[height_map[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image


def vis_kernel_height_blurred_bad():
    '''
        4*4 images:
            2*4 kernel(0~32)
            2*2 height
            2*2 blurred(according to the height)
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    fig = plt.figure(figsize = (10,8))
    
    
    
    ''' kernels '''
    for pos, (i,j) in enumerate(product([0,1,2,3], [1,2])):
        kernel = np.load('./plots_npy/color-real-3um/G-{:0>2}.npy'.format(pos*4))
        
        kernel = kernel/np.sum(kernel)
        kernel = cv2.Mat(kernel*255)
        kernel = cv2.resize(kernel,(21,21))
        kernel = kernel/np.sum(kernel)
        
        plt.subplot(4,5,5*i+j)
        # plt.imshow(kernel, cmap='gray',vmin=0,vmax=1)
        plt.imshow(kernel, cmap='gray'),plt.title("Z={}um".format(pos*4))
        plt.axis('off')
    
    
    ''' height map '''
    hs = []
    axs = []
    vs = [
        (0 ,5 ),
        (5 ,10),
        (10,15),
        (15,32),
    ]
    
    for pos,(i,j) in enumerate(product([0,1,2,3],[5])):
        vmin, vmax = vs[pos]
        height_map = get_height_map(vmin = vmin, vmax=vmax)
        hs.append(height_map)
        # print(i,j,4*i+j)
        ax = plt.subplot(4, 5, 5*i+j)
        axs.append(ax)
        im = plt.imshow(height_map, cmap='gray', vmin=0, vmax = 64)
        plt.axis('off')
        
    # plt.colorbar(im, ax=axs,fraction=0.046, )
    
    ax1 = plt.subplot(4,5,20)
    cax = fig.add_axes([ax1.get_position().x1 + 0.01 ,ax1.get_position().y0+0.05,0.02,ax1.get_position().height*4])
    plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)

    
    
    ''' blurred images '''
    images_roots = [
        r"D:\desktop\de-OOF\data\CRC-224\CRC-02-16-17-08\val-clear\ADI-TCGA-PIYMRPFC.png",
        r"D:\desktop\de-OOF\data\CRC-224\CRC-02-16-17-08\val-clear\MUS-TCGA-CLSQHIHV.png",
        r"D:\desktop\de-OOF\data\CRC-224\CRC-02-16-17-08\val-clear\MUS-TCGA-NAPEKRQQ.png",
        r"D:\desktop\de-OOF\data\CRC-224\CRC-02-16-17-08\val-clear\NORM-TCGA-REICISDL.png"
    ]
    
    for seq, (i,j) in enumerate(product([0,1,2,3], [4])):
        image = cv2.imread(images_roots[seq])
        blurred_image = blur_one_image_RGB_known_height(image, hs[seq])
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        plt.subplot(4,5,5*i+j),plt.imshow(blurred_image), plt.axis('off')
        plt.subplot(4,5,5*i+j-1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off')



    plt.subplots_adjust(
        left = 0.05,
        bottom = 0.05,
        right = 0.90,
        top = 0.95,
    )
    plt.savefig(r"D:\desktop\OOF\tmpfig.png")
    plt.show()
    
if __name__ == '__main__':
    # good_vis_kernel()
    vis_kernel_height_blurred_bad()
    ...

        