import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import time
from PIL import Image


def vis_kernel_mapping_smoothed():
    plt.figure()
    ROW = 4
    COL = 8
    for i in range(ROW*COL):
        plt.subplot(ROW, COL, i+1)
        
        kernel_mapping = np.random.randint(low = 0, high = 50, size=(224,224))
        kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
        
        kernel_mapping_smoothed = kernel_mapping.copy()
        kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=5, ksize=(0,0))
        kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=6, ksize=(0,0))
        
        kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
        
        kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
        
        tmp_m = 19 - np.max(kernel_mapping_smoothed)
        
        for _ in range(tmp_m+1):
            kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
        
        plt.imshow(kernel_mapping_smoothed), plt.axis('off')
        
        # if i == 7:
        #     plt.colorbar()
    
    plt.show()

def blur_one_image(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)
    kernel_mapping = np.random.randint(low = 0, high = 50, size=(224,224,3))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=5, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=6, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 19 - np.max(kernel_mapping_smoothed)
    for i in range(tmp_m+1):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i in range(3):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        # print(z_min, z_max)
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/zz{:.4f}.npy'.format(j/10000))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(11,11))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image

def make_dataset_with_random_spf_advanced():
    import threading
    
    TRAIN_IMAGE_NUMBER = 1024 * 4
    VALIDATION_IMAGE_NUMBER = 1024
    
    data_root = './data/CRC-224/raw'
    image_names = os.listdir(data_root)
    random.shuffle(image_names)
    
    image_names_train = image_names[:TRAIN_IMAGE_NUMBER]
    image_names_val = image_names[TRAIN_IMAGE_NUMBER:TRAIN_IMAGE_NUMBER+VALIDATION_IMAGE_NUMBER]
    
    generated_dataset_name = os.path.join('./data/CRC-224', 'CRC-'+time.strftime('%m-%d-%H-%M', time.localtime()))

    def blur_dataset_and_load(image_names, name):
        root_clear = os.path.join(generated_dataset_name, '{}-clear'.format(name))
        root_blurred = os.path.join(generated_dataset_name, '{}-blurred'.format(name))
        
        os.makedirs(root_clear, exist_ok=True)
        os.makedirs(root_blurred, exist_ok=True)
        
        with open(os.path.join(generated_dataset_name, '{}.txt'.format(name)), 'w') as f:
            f.write('\n'.join(image_names))
            
        for image_name in image_names:
            img = cv2.imread(os.path.join(data_root, image_name))
            blured_img = blur_one_image(img)
            
            cv2.imwrite(os.path.join(root_clear, image_name), img)
            cv2.imwrite(os.path.join(root_blurred, image_name), blured_img)
    ts = []
    for i in range(4):
        ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_train[1024*i:1024*(i+1)], 'train')))
        
    ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_val, 'train')))

    blur_dataset_and_load(image_names_train, 'train')
    blur_dataset_and_load(image_names_val, 'val')
    
    print('Dataset generation finished. Dataset Name:{}'.format(generated_dataset_name))
    
def make_dataset_with_random_spf():
    TRAIN_IMAGE_NUMBER = 1024*3
    VALIDATION_IMAGE_NUMBER = 1024*1
    
    data_root = './data/CRC-224/raw'
    image_names = os.listdir(data_root)
    random.shuffle(image_names)
    
    image_names_train = image_names[:TRAIN_IMAGE_NUMBER]
    image_names_val = image_names[TRAIN_IMAGE_NUMBER:TRAIN_IMAGE_NUMBER+VALIDATION_IMAGE_NUMBER]
    
    real_z = np.linspace(0,0.0019,20)
    kernels = []
    for item in real_z:
        zz = np.load('./plots_npy/zz{:.4f}.npy'.format(item))
        zz = zz/np.sum(zz)
        img = cv2.Mat(zz*255)
        img = cv2.resize(img,(7,7))
        img = img/np.sum(img)
        kernels.append(img)
        
    p = [0.005,0.005,0.005,0.005,0.005,
         0.015,0.015,0.015,0.015,0.015,
         0.090,0.090,0.090,0.090,0.090,
         0.090,0.090,0.090,0.090,0.090,]
    
    generated_dataset_name = os.path.join('./data/CRC-224', 'CRC-'+time.strftime('%m-%d-%H-%M', time.localtime()))

    def blur_dataset_and_load(image_names, name):
        root_clear = os.path.join(generated_dataset_name, '{}-clear'.format(name))
        root_blurred = os.path.join(generated_dataset_name, '{}-blurred'.format(name))
        
        os.makedirs(root_clear, exist_ok=True)
        os.makedirs(root_blurred, exist_ok=True)
        
        with open(os.path.join(generated_dataset_name, '{}.txt'.format(name)), 'w') as f:
            f.write('\n'.join(image_names))
            
        for image_name in image_names:
            kernel_index = np.random.choice(range(20),p=p)
            img = cv2.imread(os.path.join(data_root, image_name))
            blured_img = cv2.filter2D(img, 3, kernels[kernel_index])
            
            cv2.imwrite(os.path.join(root_clear, image_name), img)
            cv2.imwrite(os.path.join(root_blurred, image_name), blured_img)
        
    blur_dataset_and_load(image_names_train, 'train')
    blur_dataset_and_load(image_names_val, 'val')
            
def delete_small_images():
    data_root = r"D:/BaiduNetdiskDownload/CRC-Normal"
    image_names = os.listdir(data_root)
    
    for i, image_name in enumerate(image_names):
        size = os.path.getsize(os.path.join(data_root, image_name))
        # print(size)
        
        if size < 200_000:
            os.remove(os.path.join(data_root, image_name))
        # if i > 100:
        #     break
    
def copy_and_cut_512_to_256():
    src_root = r"D:\BaiduNetdiskDownload\CRC-Normal-Clear"
    dst_root = './data/CRC-256'
    
    cut_range = [
        [0,256,0,256],
        [0,256,256,512],
        [256,512,0,256],
        [256,512,256,512]
    ]

    for i, image_name in enumerate(os.listdir(src_root)):
        img = cv2.imread(os.path.join(src_root, image_name))
        
        for j, r in enumerate(cut_range):
            cut_img = img[r[0]:r[1], r[2]:r[3]]
            cv2.imwrite(os.path.join(dst_root, image_name.replace('.png', '_{}.png'.format(j))), cut_img)
            
        if i > 500:
            break
        
def copy_all_to_one():
    import tifffile
    from PIL import Image
    
    src_root = r"D:\BaiduNetdiskDownload\CRC-VAL-HE-7K\CRC-VAL-HE-7K"
    
    for cate in os.listdir(src_root):
        for image_name in os.listdir(os.path.join(src_root, cate)):
            image_root = os.path.join(src_root, cate, image_name)
            # image = np.array(Image.open(image_root))
            image = Image.open(image_root)
            image.save('./data/CRC-224/{}'.format(image_name.replace('.tif','.png')))
            
def make_dataset_with_random_spf_advanced_multi():
    import threading
    
    TRAIN_IMAGE_NUMBER = 1024 * 4
    VALIDATION_IMAGE_NUMBER = 1024
    
    data_root = './data/CRC-224/raw'
    image_names = os.listdir(data_root)
    random.shuffle(image_names)
    
    image_names_train = image_names[:TRAIN_IMAGE_NUMBER]
    image_names_val = image_names[TRAIN_IMAGE_NUMBER:TRAIN_IMAGE_NUMBER+VALIDATION_IMAGE_NUMBER]
    
    generated_dataset_name = os.path.join('./data/CRC-224', 'CRC-'+time.strftime('%m-%d-%H-%M', time.localtime()))

    def blur_dataset_and_load(image_names, name):
        root_clear = os.path.join(generated_dataset_name, '{}-clear'.format(name))
        root_blurred = os.path.join(generated_dataset_name, '{}-blurred'.format(name))
        
        os.makedirs(root_clear, exist_ok=True)
        os.makedirs(root_blurred, exist_ok=True)
            
        for image_name in image_names:
            img = cv2.imread(os.path.join(data_root, image_name))
            blured_img = blur_one_image_RGB(img)
            
            cv2.imwrite(os.path.join(root_clear, image_name), img)
            cv2.imwrite(os.path.join(root_blurred, image_name), blured_img)
            
    ts = []
    for i in range(4):
        ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_train[1024*i:1024*(i+1)], 'train')))
        
    ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_val, 'val')))
    
    for t in ts:
        t.start()

    # blur_dataset_and_load(image_names_train, 'train')
    # blur_dataset_and_load(image_names_val, 'val')
    
    print('Dataset generation finished. Dataset Name:{}'.format(generated_dataset_name))
    
def blur_one_image_RGB(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)
    kernel_mapping = np.random.randint(low = 0, high = 50, size=(224,224,3))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=5, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=6, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 40 - np.max(kernel_mapping_smoothed)
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.7,0.3])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        # print(z_min, z_max)
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/color/zz{:.4f}-{}.npy'.format(j/10000, name))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(11,11))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image

def blur_one_image_RGB(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)
    kernel_mapping = np.random.randint(low = 0, high = 50, size=(224,224,3))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=5, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=6, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 30 - np.max(kernel_mapping_smoothed)
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.5,0.5])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        # print(z_min, z_max)
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/color/zz{:.4f}-{}.npy'.format(j/10000, name))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(11,11))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image

def blur_one_image_RGB_real(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)
    kernel_mapping = np.random.randint(low = 0, high = 50, size=(224,224,3))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=5, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=6, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 30 - np.max(kernel_mapping_smoothed)
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.3,0.7])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(21,21))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image

def blur_one_image_RGB_real_real(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)

        
    kernel_mapping = np.random.randint(low = 0, high = 70, size=(224,224))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=8, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 30 - np.max(kernel_mapping_smoothed)
    
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed)
    
    # plt.subplot(131)
    # plt.imshow(kernel_mapping_smoothed/np.max(kernel_mapping_smoothed))
    # plt.show()
    # exit()
    
    kernel_mapping_smoothed = np.dstack([kernel_mapping_smoothed, kernel_mapping_smoothed, kernel_mapping_smoothed])
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        for j in range(z_min, z_max + 1):

            # print("j:{}".format(j))
            kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(21,21))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    result_image = np.array(result_image, dtype=np.uint8)
    
    # plt.subplot(132)
    # plt.imshow(result_image)
    # plt.subplot(133)
    # plt.imshow(ori_img)
    # plt.show()
    
    return result_image

def make_dataset_with_random_spf_advanced_multi_real():
    import threading
    
    TRAIN_IMAGE_NUMBER = 1024 * 4
    VALIDATION_IMAGE_NUMBER = 1024
    
    data_root = './data/CRC-224/raw'
    image_names = os.listdir(data_root)
    random.shuffle(image_names)
    
    image_names_train = image_names[:TRAIN_IMAGE_NUMBER]
    image_names_val = image_names[TRAIN_IMAGE_NUMBER:TRAIN_IMAGE_NUMBER+VALIDATION_IMAGE_NUMBER]
    
    generated_dataset_name = os.path.join('./data/CRC-224', 'CRC-'+time.strftime('%m-%d-%H-%M', time.localtime()))

    def blur_dataset_and_load(image_names, name):
        root_clear = os.path.join(generated_dataset_name, '{}-clear'.format(name))
        root_blurred = os.path.join(generated_dataset_name, '{}-blurred'.format(name))
        
        os.makedirs(root_clear, exist_ok=True)
        os.makedirs(root_blurred, exist_ok=True)
            
        for image_name in image_names:
            img = cv2.imread(os.path.join(data_root, image_name))
            blured_img = blur_one_image_RGB_real_real(img)
            
            cv2.imwrite(os.path.join(root_clear, image_name), img)
            cv2.imwrite(os.path.join(root_blurred, image_name), blured_img)
            
    ts = []
    for i in range(4):
        ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_train[1024*i:1024*(i+1)], 'train')))
        
    ts.append(threading.Thread(target=blur_dataset_and_load, args=(image_names_val, 'val')))
    
    for t in ts:
        t.start()

    for t in ts:
        t.join()
        
    # blur_dataset_and_load(image_names_train, 'train')
    # blur_dataset_and_load(image_names_val, 'val')
    
    print('Dataset generation finished. Dataset Name:{}'.format(generated_dataset_name))
    
if __name__ == '__main__':
    # make_dataset_with_random_spf()
    # img = cv2.imread('./data/CRC-224/raw/ADI-TCGA-AAICEQFN.png')
    # blur_one_image(img)
    # make_dataset_with_random_spf_advanced()
    make_dataset_with_random_spf_advanced_multi_real()
    # blur_one_image_RGB_real_real(cv2.imread('./data/CRC-224/raw/TUM-TCGA-YYKLKLPC.png'))
    # blur_one_image()
    # vis_kernel_mapping_smoothed()
    # raw = np.zeros((3,3,2))
    
    # mapping = [
    #     [1,2,1],
    #     [2,4,2],
    #     [1,2,1],
    # ]
    # mapping = np.array(mapping)
    
    # raw[mapping == 1,0]  = raw [mapping == 1,0] +5 
    # raw[mapping == 2,0]  = raw [mapping == 2,0] +10
    # raw[mapping == 4,0]  = raw [mapping == 4,0] +15
    
    # raw[mapping == 1,1]  = raw [mapping == 1,1] +6
    # raw[mapping == 2,1]  = raw [mapping == 2,1] +7
    # raw[mapping == 4,1]  = raw [mapping == 4,1] +8
    
    # print(raw)
    
    
    