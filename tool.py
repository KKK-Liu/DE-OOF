import os
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import sqrt, power
from itertools import product


def blurall():
    image_root = './data/CRC-1k/clear-100'
    image_names = os.listdir(image_root)
    print(len(image_names))

    kernelg33 = np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],dtype=np.float64
    )/16
    print(np.sum(kernelg33))
    print(kernelg33)

    for image_name in image_names:
        image_load_root = os.path.join(image_root, image_name)
        img = np.array(Image.open(image_load_root))


        # blured_img = cv2.filter2D(img, 3, kernelg33)
        blured_img1 = cv2.GaussianBlur(img, None, sigmaX=1,sigmaY=1)
        blured_img2 = cv2.GaussianBlur(img, None, sigmaX=2,sigmaY=2)
        blured_img3 = cv2.GaussianBlur(img, None, sigmaX=3,sigmaY=3)
        
        
        plt.subplot(241),plt.imshow(img), plt.axis('off'),plt.title('raw')
        plt.subplot(242),plt.imshow(blured_img1), plt.axis('off'),plt.title('sigma=1.0')
        plt.subplot(243),plt.imshow(blured_img2), plt.axis('off'),plt.title('sigma=2.0')
        plt.subplot(244),plt.imshow(blured_img3), plt.axis('off'),plt.title('sigma=3.0')
        
        
        plt.show()
        break

def distanceL2(x,y,a,b):
    return sqrt(pow(x-a,2)+pow(y-b,2))


def airyPattern():
    from scipy import special
    from itertools import product
    # x=np.linspace(-10,10,21)
    # X,Y = np.meshgrid(x,x)
    # # print(mat)
    # print(x)
    # plt.plot(X,Y,linestyle='',marker='.')
    # plt.show()
    I0 = 2
    mat = np.zeros((101,101),dtype=np.float64)
    for i,j in product(range(101), range(101)):
        r = distanceL2(i,j,50,50)
        mat[i,j] = I0 * pow(special.j1(r)/r,2)
    
    # x = np.linspace(-10,10,100)
    # y = special.j0(x)
    # plt.plot(x,y)
    # plt.show()
    plt.imshow(mat)
    plt.show()
    
    # print(special.jv(1,1))
    
def airyPattern1():
    from scipy import special
    from itertools import product
    L = 11
    C = int(L/2)
    R = 8
    I0 = 1
    mat = np.zeros((L,L),dtype=np.float64)
    
    for i,j in product(range(L), range(L)):
        r = distanceL2(i,j,C,C)/L*2*R
        if i==C and j==C:
            mat[i,j] = I0 * 0.25
        else:
            mat[i,j] = I0 * pow(special.j1(r)/r,2)
    mat = mat/np.sum(mat)
    print(np.sum(mat))
    # plt.imshow(mat)
    # plt.colorbar()
    # plt.show()
    return mat
    
def airycurve():
    from scipy import special
    from itertools import product
    # I0 = 2
    
    x = np.linspace(-20,20,1000)
    y = np.power(special.j1(x)/x,2)
    
    plt.plot(x,y)
    
    plt.show()
    
def Gkernel():
    from scipy import special
    from itertools import product
    from math import exp
    sigma = 5
    
    L = 11
    C = int(L/2)
    R = 8
    I0 = 1
    mat = np.zeros((L,L),dtype=np.float64)
    
    for i,j in product(range(L), range(L)):
        r = distanceL2(i,j,C,C)/L*2*R
        mat[i,j] = exp(-power(r/sigma,2)/2)/sigma/sqrt(2*np.pi)
    mat = mat/np.sum(mat)
    print(np.sum(mat))
    # plt.imshow(mat)
    # plt.colorbar()
    # plt.show()
    return mat
    
    
def cmpakgk():
    Gk = Gkernel()
    Ak = airyPattern1()

    # p = np.hstack((Gk,Ak))

    plt.subplot(1,2,1),plt.imshow(Gk), plt.title("gk"),plt.colorbar()
    plt.subplot(1,2,2),plt.imshow(Ak), plt.title("ak"),plt.colorbar()
    # plt.imshow(p)
    # plt.colorbar()
    plt.show()


def blurone():
    image_root = './data/CRC'
    
    row = 8
    col = 12
    
    image_names = os.listdir(image_root)
    print(len(image_names))

    for image_name in image_names:
        image_load_root = os.path.join(image_root, image_name)
        img = np.array(Image.open(image_load_root))

        for i,j in product(range(row), range(col)):
            blured_img = cv2.GaussianBlur(img, None, sigmaX=0.4*j+0.01,sigmaY=0.4*i+0.01)
            plt.subplot(row,col,col*i+j+1),plt.imshow(blured_img),plt.axis('off')
        # blured_img = cv2.filter2D(img, 3, kernelg33)
        # blured_img1 = cv2.GaussianBlur(img, None, sigmaX=1,sigmaY=1)
        # blured_img2 = cv2.GaussianBlur(img, None, sigmaX=2,sigmaY=2)
        # blured_img3 = cv2.GaussianBlur(img, None, sigmaX=3,sigmaY=3)
        
        
        # plt.subplot(141),plt.imshow(img), plt.axis('off'),plt.title('raw')
        # plt.subplot(142),plt.imshow(blured_img1), plt.axis('off'),plt.title('sigma=1.0')
        # plt.subplot(143),plt.imshow(blured_img2), plt.axis('off'),plt.title('sigma=2.0')
        # plt.subplot(144),plt.imshow(blured_img3), plt.axis('off'),plt.title('sigma=3.0')
        plt.subplot(row,col,1),plt.imshow(img),plt.axis('off')
        plt.subplots_adjust(0,0,1,1,0,0)
        # plt.tight_layout()
        plt.show()
        break
    
def makeDataset():
    from random import shuffle, random
    import shutil
    from PIL import Image
    import numpy as np
    
    # sigma range should be [1,2]
    
    oriroot = './data/CRC/'
    image_names = os.listdir(oriroot)
    
    shuffle(image_names)
    
    image_num = 1000
    
    clear_root = './data/CRC-1k/clear'
    blur_unpaired = './data/CRC-1k/blur-unpaired'
    blur_paired = './data/CRC-1k/blur-paired'
    
    os.makedirs(clear_root, exist_ok=True)
    os.makedirs(blur_unpaired, exist_ok=True)
    os.makedirs(blur_paired, exist_ok=True)
    
    for image_name in image_names[0:image_num]:
        shutil.copy(
            src = os.path.join(oriroot, image_name),
            dst = os.path.join(clear_root, image_name)
        )
        
    for image_name in image_names[0:image_num]:
        img = np.array(Image.open(os.path.join(oriroot, image_name)))
        sigmaX = 1 + random()
        sigmaY = 1 + random()
        blurred_img = cv2.GaussianBlur(img, None,sigmaX=sigmaX, sigmaY=sigmaY)
        Image.fromarray(blurred_img).save(os.path.join(blur_paired, image_name))
        
    for image_name in image_names[image_num: 2*image_num]:
        img = np.array(Image.open(os.path.join(oriroot, image_name)))
        sigmaX = 1 + random()
        sigmaY = 1 + random()
        blurred_img = cv2.GaussianBlur(img, None,sigmaX=sigmaX, sigmaY=sigmaY)
        Image.fromarray(blurred_img).save(os.path.join(blur_unpaired, image_name))
    
def f(x):
    from scipy import special
    lam = 500*10**(-6)
    k=2*np.pi/lam
    NA = 1.5
    ni = 1.5
    
    # return special.j0(k*NA*)

def ff(x,y):
    return x*y

def real_f(rou,r,z):
    from scipy import special
    lam = 500*10**(-6)
    k=2*np.pi/lam
    NA = 1.5
    ni = 1.5
    
    return special.j0(k*NA*r*rou/ni)*np.cos(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou

def image_f(rou,r,z):
    from scipy import special
    lam = 500*10**(-6)
    k=2*np.pi/lam
    NA = 1.5
    ni = 1.5
    
    return special.j0(k*NA*r*rou/ni)*np.sin(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou
    

def real_part(r,z):
    from scipy import integrate
    
    return integrate.tplquad(real_f,0,1,0,r,0,z)[0]

    
def image_part(r,z):
    from scipy import integrate
    
    return integrate.tplquad(image_f,0,1,0,r,0,z)[0]

def hrz(r,z):
    return np.sqrt(np.power(real_part(r,z),2)+np.power(image_part(r,z),2))
    

def hxzy(x,y,z, lam):
    from scipy import integrate
    from scipy import special
    
    r = np.sqrt(np.power(x,2)+np.power(y,2))
    
    k = 2*np.pi/lam
    NA = 1.5
    ni = 1.5
            
    def hrz():
        def real_f(rou):
            return special.j0(k*NA*r*rou/ni)*np.cos(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou

        def image_f(rou):
            return special.j0(k*NA*r*rou/ni)*np.sin(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou 
        
        def real_part():
            return integrate.quad(real_f,0,1)[0]
        
        def image_part():
            return integrate.quad(image_f,0,1)[0]
        
        if z == 0:
            return np.power(real_part(),2)
        else:
            return np.power(real_part(),2)+np.power(image_part(),2)
    
    return hrz()

def psf_map_generate_z0xy():
    from itertools import product
    num = 101
    low = -0.0003
    high = 0.0003
    
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    z = np.zeros(num)
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[j],z[i])
    
    plt.imshow(zz)
    plt.show() 
    ...
    
def psf_line_generate():
    num = 101
    low = -0.0003
    high = 0.0003
    x = np.linspace(low,high,num) 
    y = np.zeros(num) 
    z = np.zeros(num)
    
    
    # zz = hxzy(x,y,z)
    zz = np.zeros(num)
    for i in range(num):
        zz[i] = hxzy(x[i],y[i],z[i])
    
    plt.plot(x,zz)
    plt.show() 
    
def psf_map_generate_x4zy():
    from itertools import product
    num = 101
    low = -0.001
    high = 0.001
    x = np.ones(num)*0.0004
    y = np.linspace(low,high,num) 
    z = np.linspace(low,high,num) 
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[i],z[j])
    
    plt.imshow(zz, cmap='gray')
    plt.show() 
    
def psf_map_generate_z0xy_3d():
    from itertools import product
    num = 101
    low = -0.0003
    high = 0.0003
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    z = np.zeros(num)
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[j],z[i])
    
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, zz,cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()
    
def psf_map_generate_x0yz():
    from itertools import product
    num = 101
    low = -0.001
    high = 0.001
    x = np.zeros(num)
    y = np.linspace(low,high,num) 
    z = np.linspace(low,high,num) 
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[i],z[j])
    
    plt.imshow(zz)
    plt.show() 
    np.save('./x0yz.npy',zz)
    
def load_npy_and_show():
    zz = np.load('./x0yz.npy')
    plt.imshow(zz)
    plt.show()
    
def psf_map_generate_zxy_3d():
    from itertools import product
    num = 101
    low = -0.001
    high = 0.001
    x = np.linspace(low,high,num)
    y = np.linspace(low,high,num)
    z = np.ones(num) * -0.0000
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[j],z[i])
    
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, zz,cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()
    
def psf_map_generate_zxy_manyz():
    from itertools import product
    real_z = np.linspace(-0.001,0.001,11)
    os.makedirs('./plots_npy',exist_ok=True)
    for item in real_z:
        num = 101
        low = -0.001
        high = 0.001
        x = np.linspace(low,high,num) 
        y = np.linspace(low,high,num) 
        z = np.ones(num) * item
        
        zz = np.zeros((num,num))
        for i,j in product(range(num), range(num)):
            zz[i,j] = hxzy(x[i],y[j],z[i])
        
        np.save('./plots_npy/zz{:.4f}.npy'.format(item), zz)
        # X, Y = np.meshgrid(x, y)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X, Y, zz,cmap='viridis', edgecolor='none')

    # plt.show()
    
def psf_map_generate_zxy_load_many():
    from itertools import product
    real_z = np.linspace(-0.001,0.001,11)
    # os.makedirs('./plots_npy',exist_ok=True)
    plt.figure()
    for i, item in enumerate(real_z):
        zz = np.load('./plots_npy/zz{:.4f}.npy'.format(item))
        plt.subplot(2,6,i+1),plt.imshow(zz),plt.title("{:.4f}".format(item)),plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def psf_map_generate_zxy_positivez():
    from itertools import product
    real_z = np.linspace(0,0.0040,41)

    os.makedirs('./plots_npy/color/',exist_ok=True)
    print('make diretory')
    # R:700.0, G:546.1, B:435.8 nm 
    for lam, name in zip([700, 546.1, 435.8], ['R', 'G', 'B']):
        lam = lam * 10**(-6)
        for item in real_z:
            num = 101
            low = -0.001
            high = 0.001
            x = np.linspace(low,high,num) 
            y = np.linspace(low,high,num) 
            z = np.ones(num) * item
            
            zz = np.zeros((num,num))
            for i,j in product(range(num), range(num)):
                zz[i,j] = hxzy(x[i],y[j],z[i], lam)

            file_name = './plots_npy/color/zz{:.4f}-{}.npy'.format(item, name)
            np.save(file_name, zz)
            print(file_name)
            

        
def psf_map_generate_zxy_load_positivez():
    from itertools import product
    real_z = np.linspace(0,0.0017,18)
    for i, item in enumerate(real_z):
        zz = np.load('./plots_npy/zz{:.4f}.npy'.format(item))
        
        zz = zz/np.sum(zz)
        print(np.sum(zz))
        plt.subplot(3,6,i+1),plt.imshow(zz),plt.title("{:.4f}".format(item)),plt.axis('off')
    plt.tight_layout()
    plt.show()
        
        
def psf_map_generate_zxy_load_positivez_and_downSample():
    from itertools import product
    real_z = np.linspace(0,0.0017,18)
    for i, item in enumerate(real_z):
        zz = np.load('./plots_npy/zz{:.4f}.npy'.format(item))
        
        zz = zz/np.sum(zz)
        
        img = cv2.Mat(zz*255)
        img = cv2.resize(img,(7,7))
        img = img/np.sum(img)
        # print(np.sum(zz))
        plt.subplot(3,6,i+1),plt.imshow(img),plt.title("{:.4f}".format(item)),plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def blur_with_psf_kernel():
    
    data_root = './data/CRC'
    image_names = os.listdir(data_root)
    
    img_name = os.path.join(data_root, image_names[10])
    img = Image.open(img_name)
    
    img = np.array(img)
    z = np.linspace(0,0.0017,18)
    for i, item in enumerate(z):
        kernel = np.load('./plots_npy/zz{:.4f}.npy'.format(item))
        
        kernel = kernel/np.sum(kernel)
        
        kernel = cv2.Mat(kernel*255)
        kernel = cv2.resize(kernel,(7,7))
        kernel = kernel/np.sum(kernel)
        
        blured_img = cv2.filter2D(img,3,kernel=kernel)
        
        plt.subplot(3,6,i+1),plt.imshow(blured_img),plt.title("{:.4f}".format(item)),plt.axis('off')
        
        
    plt.tight_layout()
    plt.show()
    
    
def psf_map_generate_zxy_positivez_show():
    from itertools import product
    # real_z = np.linspace(0,0.0040,41)
    # R:700.0, G:546.1, B:435.8 nm 
    lam = 500 * 10 ** (-9)

    num = 101
    low = -5.25 * 10 **(-6)
    high = 5.25 * 10 **(-6)
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    z = np.ones(num) * 2 * 10 **(-6)
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[j],z[i], lam)

    plt.imshow(zz)
    plt.show()
    
    
def psf_map_generate_zxy_positivez_real_one():
    
    from itertools import product
    # real_z = np.linspace(0,0.0040,41)
    # R:700.0, G:546.1, B:435.8 nm 
    lam = 500 * 10 ** (-9)

    num = 101
    low = -5.25 * 10 **(-6)
    high = 5.25 * 10 **(-6)
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    z = np.ones(num) * 2 * 10 **(-6)
    
    zz = np.zeros((num,num))
    for i,j in product(range(num), range(num)):
        zz[i,j] = hxzy(x[i],y[j],z[i], lam)

    plt.imshow(zz)
    plt.show()

def psf_map_generate_zxy_positivez_real():
    from itertools import product
    real_z = np.linspace(0,63,64)
    save_root = './plots_npy/color-real-3um/'
    os.makedirs(save_root,exist_ok=True)
    print('make diretory')
    # R:700.0, G:546.1, B:435.8 nm 
    for lam, name in zip([700, 546.1, 435.8], ['R', 'G', 'B']):
        lam = lam * 10 ** (-9)
        for item in real_z:
            item_z = item * 10 **(-7)
            
            num = 101
            low = -5.25 * 10 **(-6)
            high = 5.25 * 10 **(-6)
            
            x = np.linspace(low,high,num) 
            y = np.linspace(low,high,num) 
            z = np.ones(num) * item_z
            
            zz = np.zeros((num,num))
            
            for i,j in product(range(num), range(num)):
                zz[i,j] = hxzy(x[i],y[j],z[i], lam)

            file_name = os.path.join(save_root, '{}-{:0>2}.npy'.format(name, int(item)))
            
            np.save(file_name, zz)
            print(file_name)
            
def calculate(name,batch_data):
        print(name,' start')
        save_root = './plots_npy/color-real-3um/'
        low = -5.25 * 10 **(-6)
        high = 5.25 * 10 **(-6)
        num = 101
        x = np.linspace(low,high,num) 
        y = np.linspace(low,high,num) 
        
        for item, (lam, name) in batch_data:
            lam = lam * 10 ** (-9)
            item_z = item * 10 **(-7)
            z = np.ones(num) * item_z
            zz = np.zeros((num,num))
            
            for i,j in product(range(num), range(num)):
                zz[i,j] = hxzy(x[i],y[j],z[i], lam)

            file_name = os.path.join(save_root, '{}-{:0>2}.npy'.format(name, int(item)))
            
            np.save(file_name, zz)
            print(file_name)
        print(name,' finish')
        
def psf_map_generate_zxy_positivez_real_multi_thread():
    from itertools import product
    from multiprocessing import Process
    # from threading import Thread
    NUM_OF_THREAD =12
    
    save_root = './plots_npy/color-real-3um/'
    os.makedirs(save_root,exist_ok=True)
    
    print('make diretory')
    
    low = -5.25 * 10 **(-6)
    high = 5.25 * 10 **(-6)
    num = 101
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    # R:700.0, G:546.1, B:435.8 nm 
    all_items = list(product(np.linspace(0,63,64),zip([700, 546.1, 435.8], ['R', 'G', 'B'])))
    
    step = int(len(all_items)/NUM_OF_THREAD)
    
    all_items_batches = [all_items[i:i+step] for i in range(0,len(all_items), step)]
    print(len(all_items_batches),' batches')
    
    ts = [Process(target=calculate, args=(i, all_items_batches[i],)) for i in range(NUM_OF_THREAD)]
    
    [t.start() for t in ts]
    [t.join() for t in ts]
    

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

def flo():
    # item = 5
    for item in range(31):
        print('{:0>2}'.format(item))
        
def blur_one_image_RGB_real(ori_img=None):
    '''
        generate the mapping relation from pixel to kernel
    '''
    # if ori_img != None:
    img = np.array(ori_img)
    # print(img.shape)
    # img = cv2.resize(img, (224,224))
    # print(img.shape)
    
    # print(1)
        
    kernel_mapping = np.random.randint(low = 0, high = 100, size=(224,224))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=2, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 50 - np.max(kernel_mapping_smoothed)
    
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.3,0.7])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed) * 5
    
    plt.subplot(121)
    plt.imshow(kernel_mapping_smoothed/np.max(kernel_mapping_smoothed))
    # kernel_mapping_smoothed = kernel_mapping_smoothed/np.max(kernel_mapping_smoothed)
    
    
    # kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    # plt.imshow(kernel_mapping_smoothed[:,:,0])
    kernel_mapping_smoothed = np.dstack([kernel_mapping_smoothed, kernel_mapping_smoothed, kernel_mapping_smoothed])
    
    print(kernel_mapping_smoothed.shape)
    
    # plt.show()
    # return

    print(img.shape)
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
    
    plt.subplot(122)
    plt.imshow(result_image)
    # plt.show()
    return result_image, kernel_mapping_smoothed



def make_zip(source_dir:str, output_filename):
    import zipfile
    
    ignored_dirlist = [
        'data',
        'logs',
        'results',
        'ref-codes',
        'wandb',
        '.git',
        'TmpImages',
        '__pycache__',
        'checkpoints',
    ]
    
    # ignored_dirlist = [
    #     'p3'
    # ]
    source_dir = source_dir.replace('/','\\')
    
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        # parent:str
        # if os.path.split(parent)[1] in ignored_dirlist:
        #     continue
        f = False
        for ignore_dir in ignored_dirlist:
            if '\\{}\\'.format(ignore_dir) in parent:
                f = True
                break
            if parent.endswith('\\{}'.format(ignore_dir)):
                f = True
                break
        if f:
            continue
        for filename in filenames:
            # if filename == output_filename:
            #     continue
            
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # ????????????
            
            zipf.write(pathfile, arcname)
            print(arcname)
        # print()
    zipf.close()

def oswalk_test():
    import os
    
    for parent, dirnames, filenames in os.walk("D:\\desktop\\f"):
        print('-'*10)
        print('parent',parent)
        print('dirnames', dirnames)
        print('filenames',filenames)
        
    
def mse2psnr():
    from math import log10
    # mse = float(input('MSE:'))
    mse = 0.0003782
    
    psnr = -10 * log10(mse)
    
    print('psnr:{}'.format( psnr))


def wandb_test():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='this is name')
    parser.add_argument('--epoch', type=int, default = 200)
    
    args= parser.parse_args()
    
    import wandb 
    
    wandb.init(
        name = args.name,
        entity = 'kkk06',
        project = 'Deblur',
        config = vars(args)
    )
    
    for i in range(100):
        this_dict = {
            'i':i,
            'i^2':i*i,
        }
        wandb.log(this_dict)
        
        
def nan_tensor_test():
    import torch
    t = torch.ones((10,10))
    t = t/0.0
    print(torch.isnan(t).any(), torch.isinf(t).any())
    
def save_result_test():
    import torchvision.transforms.functional as f
    import torch
    
    img = torch.rand((3,224,224))
    img = f.to_pil_image(img)
    
    img.save('./TmpImages/test.png')

def gray_image_generation():
    img = np.ones((200,400,3),dtype=np.uint8) * 128
    cv2.imwrite('./TmpImages/gray.png', img)
    
def nvidia_smi_test():
    lines = os.popen('nvidia-smi').read().count('0%')
    print(lines)
    
    
def rand1_test():
    import torch
    
    t = torch.rand((1),dtype=torch.float32,requires_grad=True)
    print(t.shape)
    
def generate_datalist():
    with open('./datalist_gopro.txt', 'w') as f:
        for image_name in os.listdir("D:/desktop/GOPRO_Large/train/4096images/sharp"):
            f.write('/root/autodl-tmp/data/GOPRO_Large/train/4096images/sharp/{} '.format(image_name))
            f.write('/root/autodl-tmp/data/GOPRO_Large/train/4096images/blur_gamma/{}\n'.format(image_name))
            
    with open('./train_gopro_gamma.list', 'w') as f:
        for image_name in os.listdir("D:/desktop/GOPRO_Large/train/4096images/sharp"):
            f.write('/root/autodl-tmp/data/GOPRO_Large/train/4096images/blur_gamma/{} '.format(image_name))
            f.write('/root/autodl-tmp/data/GOPRO_Large/train/4096images/sharp/{}\n'.format(image_name))
            
    with open('./val_gopro_gamma.list', 'w') as f:
        for image_name in os.listdir("D:/desktop/GOPRO_Large/test/1024images/sharp"):
            f.write('/root/autodl-tmp/data/GOPRO_Large/test/1024images/blur_gamma/{} '.format(image_name))
            f.write('/root/autodl-tmp/data/GOPRO_Large/test/1024images/sharp/{}\n'.format(image_name))
                                     
def r():
    d = 1
    print(np.pi * d * d /4)
    
def clamp_test():
    import torch
    
    t = torch.rand((3,3))*10
    print(t)
    t = t.clamp(2,3)
    print(t)
    
def nan_test_1():
    import torch
    
    t = torch.ones((5))
    print(t)
    t = t/0
    print(t)
    tt = torch.ones((5))
    print(tt)
    tt = tt + 0.0 * t
    print(tt)
    new_tt = torch.ones((5)) + 0.0 * tt
    print(new_tt)
    
def cal_mean_and_var():
    src_root = ''
    ...
    
def motion_blur():
    image_root = './data/CRC-224/raw/ADI-TCGA-AAICEQFN.png'
    # image_BGR = cv2.cvtColor(cv2.imread(image_root), cv2.COLOR_BGR2RGB)
    # # image_blurred = cv2.
    # kernel = [
    #     [1,1,1,1,1]
    # ]
    img = cv2.imread(image_root)
    img_blurred, kernel = motion_blur_copy(img)
    print(kernel.shape)
    cv2.imshow('sharp',img)
    cv2.imshow('blurred',img_blurred)
    # cv2.imshow('kernel',kernel)
    plt.imshow(kernel)
    plt.show()
    # cv2.waitKey(0)
    
def motion_blur_copy(image, degree=24, angle=30):
    image = np.array(image)

    # ???????????????????????????????????????kernel???????????? degree???????????????????????????
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle + 45, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    
    # print(motion_blur_kernel)
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred,motion_blur_kernel

def list_test_1():
    l1 = [1,2,3]
    l2 = [4,5,6]
    print(l1+l2)
    
def wsi_blur_and_cut():
    # img = np.array(ori_img)
    import openslide
    
    slide_root = r"D:/desktop/II1952204 - 2020-10-14 20.55.46.ndpi"
    STEP = 224
    with openslide.OpenSlide(slide_root) as slide:
        print('slide.dimension:', slide.dimensions)
        print('slide.level_count:', slide.level_count)
        print('slide.level_downsamples:', slide.level_downsamples)
        
        slide_w, slide_h = slide.dimensions
        
        # slide_level0 = slide.read_region((0,0),0,(slide_w, slide_h))
        # print(slide_level0.size)
        tmp_level = 8
        downsampled_slide = slide.read_region((0,0),tmp_level,(int(slide_w/slide.level_downsamples[tmp_level]), int(slide_h/slide.level_downsamples[tmp_level])))
        
        # downsampled_slide.show('level 4')
        slide_array = np.array(downsampled_slide)
        
        print(slide_array.shape)
        print(slide_array)
        
        slide_array_without_4thchannel = slide_array[:,:,:3]
        print(slide_array_without_4thchannel.shape)
        print(slide_array_without_4thchannel)
        
        plt.imshow(slide_array_without_4thchannel)
        plt.show()
        # for x,y in product(range(0, slide_w, STEP), range(0, slide_h, STEP)):
            
        
        
    exit()
    H = 224
    W = 224
        
    # mg = np.array(ori_img)

        
    kernel_mapping = np.random.randint(low = 0, high = 256, size=(H,W))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=8, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 30 - np.max(kernel_mapping_smoothed)
    
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.2,0.8])
    kernel_mapping_smoothed = np.abs(kernel_mapping_smoothed, dtype=np.float32)
    kernel_mapping_smoothed = cv2.resize(cv2.Mat(kernel_mapping_smoothed),dsize=(4096,4096))
    # plt.subplot(131)
    
    # kernel_mapping_smoothed = np.dstack([kernel_mapping_smoothed, kernel_mapping_smoothed, kernel_mapping_smoothed])
    
    # result_image = np.zeros(img.shape, dtype=np.uint8)
    
    # for i, name in zip(range(3), ['B', 'G', 'R']):
    #     z_min = np.min(kernel_mapping_smoothed[:,:,i])
    #     z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
    #     for j in range(z_min, z_max + 1):

    #         # print("j:{}".format(j))
    #         kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
    #         kernel = kernel/np.sum(kernel)
    #         kernel = cv2.Mat(kernel*255)
    #         kernel = cv2.resize(kernel,(21,21))
    #         kernel = kernel/np.sum(kernel)
            
    #         blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
    #         blurred_one_channel = np.array(blurred_one_channel)
            
    #         result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    
    # result_image = np.array(result_image, dtype=np.uint8)
    
    
'''
    ????????????????????????:
        1. blur and deblur ????????? ??????,blur,?????????, ????????????um, ????????????????????????
        2. WSI level???deblur and Attention map
'''

def blur_and_deblur_visualization():
    '''
        ??????????????????????????????
    '''
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch
    
    image_roots = {
        'raw':r"D:\desktop\de-OOF\data\CRC-224\CRC-02-26-19-56\val-clear\ADI-TCGA-AAWDNKDK.png",
        'blurred':r"D:\desktop\de-OOF\data\CRC-224\CRC-02-26-19-56\val-blurred\ADI-TCGA-AAWDNKDK.png",
        'restored':r"D:\desktop\de-OOF\data\CRC-224\CRC-02-26-19-56\val-clear\ADI-TCGA-AAWDNKDK.png",
    }

    image = {}
    
    for key in image_roots.keys():
        image[key] = np.array(Image.open(image_roots[key]))
    
    pos_x, pos_y = 0, 0
    heigth, width = 32, 32
    
    
    sub_image = {}
    
    for key in image.keys():
        sub_image[key] = image[key][pos_x:pos_x+heigth,pos_y:pos_y+width]
        
    # for i, key in enumerate(image_roots.keys()):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(image[key])
    #     plt.axis('off')
    #     plt.subplot(2,3,i+1+3)
    #     plt.imshow(sub_image[key])
    #     plt.axis('off')
        
        
    # plt.show()
    # plt.figure(figsize=(9,6))
    fig, ax = plt.subplots(2,3)
    fig.set_figheight(6)
    fig.set_figwidth(9)
    for i, key in enumerate(image_roots.keys()):
        ax[0][i].imshow(image[key])
        ax[0][i].axis('off')
        ax[1][i].axis('off')
        
        small_ax = inset_axes(ax[0][i], width="40%", height="40%", loc=3,
                   bbox_to_anchor=(0.5, 0.5, 0.8, 0.8),
                   bbox_transform=ax[0][i].transAxes)
        
        small_ax.imshow(sub_image[key])
        small_ax.axis('off')
        
        
        # box of small ax
        small_ax.plot(
            [0,32,32,0,0],
            [0,0,32,32,0],
            'blue'
        )
        
        # box of big ax
        ax[0][i].plot(
            [0,32,32,0,0],
            [0,0,32,32,0],
            'red'
        )
        
        con = ConnectionPatch(xyA=(0,32),xyB=(0,32),coordsA="data",coordsB="data",
        axesA=small_ax,axesB=ax[0][i])

        small_ax.add_artist(con)
        
        con = ConnectionPatch(xyA=(32,0),xyB=(32,0),coordsA="data",coordsB="data",
        axesA=small_ax,axesB=ax[0][i])

        small_ax.add_artist(con)
        
        # ax[0][i].indicate_inset_zoom(small_ax)
        
    plt.show()
        
        
        
    
    
    
if __name__ == '__main__':
    # generate_datalist()
    # wsi_blur_and_cut()
    blur_and_deblur_visualization()
    # r()
    # mse2psnr()
    # clamp_test()
    # psf_show()
    # nan_test_1()
    # motion_blur()
    # list_test_1()
