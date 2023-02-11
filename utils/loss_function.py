import torch.nn as nn
import torch.nn.functional as F


class loss_Function_unet(nn.Module):
    def __init__(self, reduction = 'mean') -> None:
        super().__init__()
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, x, y):
        return self.mse_loss(x, y)
    
class loss_Function_srn(nn.Module):
    def __init__(self, reduction = 'mean') -> None:
        super().__init__()
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, x, y):
        return self.mse_loss(x, y)

class loss_Function_GAN(nn.Module):
    def __init__(self, reduction = 'mean') -> None:
        super().__init__()
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, x, y):
        return self.mse_loss(x, y)
    
def get_loss_function(args):
    if args.model == 'U_Net':
        return loss_Function_unet()
    elif args.model == 'SRN-DeblurNet':
        return loss_Function_srn()
    elif args.model == 'DeblurGAN':
        return loss_Function_GAN()
    else:
        raise NotImplementedError('There is no corresponding loss function for model {}'.format(args.model))
    
