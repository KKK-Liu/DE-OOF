import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

with h5py.File('./data/outoffocus2017_patches5Classification.h5','r') as f:
    keys_list = list(f.keys())
    print(keys_list)
    print(len(keys_list))
    
    X = f["X"]
    Y = f["Y"]
    
    print(X.shape)
    print(Y.shape)
    # for i in range(200):
    #     plt.subplot(10,20,i+1),plt.imshow(X[i]),plt.axis('off'),plt.title(str(Y[i]))
    #     # print(Y[i])
    # plt.subplots_adjust(
    #     0,0,1,0.95,0.2,0.2
    # )
    # plt.show()
    # img = np.array(X[0]*255,dtype=np.uint8)
    # img = Image.fromarray(img)
    # img.save('./data/outoffocus2017/{}.png'.format(Y[0]))
    # plt.imshow(img)
    # plt.show()
    # print(X[0])
    name_cnt={}
    for i in range(12):
        os.makedirs('./data/outoffocus2017/{}.0'.format(i), exist_ok=True)
        name_cnt['{}.0'.format(i)]=0
        
    # print(name_cnt)
    
    # name_cnt['1.0'] = name_cnt['1.0']+1
        
    # print(name_cnt)
    # for img, name in tqdm(zip(X,Y)):
    #     # name = str(name)
    #     # Image.fromarray(np.array(img*255, dtype=np.uint8)).save('./data/outoffocus2017/{}/{}.png'.format(name,name_cnt[name]))
    #     name_cnt[name] = name_cnt[name] + 1
    #     # break
    
    for name in tqdm(Y):
        name = str(name)
        name_cnt[name] = name_cnt[name] + 1
    print(name_cnt)
        
        
    
    
