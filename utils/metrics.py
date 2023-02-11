from piq import psnr, ssim, multi_scale_ssim, fsim, gmsd, multi_scale_gmsd, LPIPS, PieAPP,DISTS
import torch

def metric_psnr():
    x = torch.randn(2,3, 224,224)
    y = torch.randn(2,3, 224,224)
    z = x
    
    m = DISTS()
    # m = PieAPP()
    
    print(m(x,y))
    print(m(x,z))
    
metric_psnr()