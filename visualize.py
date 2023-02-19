import matplotlib.pyplot as plt
import os
from itertools import product
from PIL import Image

def comparison_graph():
    root = ''
    image_names = []
    
    
def vis_batch():
    sharp_image_root = ''
    image_names = os.listdir(sharp_image_root)
    image_roots = [
        ('Blurred',''),
        ('Ground Truth',''),
        ('Ours',''),
        ('CycleGAN',''),
        ('DeepDeblur',''),
        ('SRN-DeblurNet',''),
        ('DeblurGAN-v2',''),
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
            
    
    
    