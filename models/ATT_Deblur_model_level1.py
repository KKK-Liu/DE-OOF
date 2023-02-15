import torch.utils.data
import torch

from torch.cuda.amp import autocast as autocast

from .ATT_Deblur_model import ATT_Deblur_Net


class ATT_Deblur_Net_level1(ATT_Deblur_Net):
    def __init__(self, args) -> None:
        super(ATT_Deblur_Net_level1, self).__init__()
        ATT_Deblur_Net.__init__(args, level=1)
        
        if self.isTrain:
            self.loss_names += ['l1_1']
            self.meter_init()


        print('SRNATTS_Net level1  is created')
        
    def train_step(self):
        self.optimizer.zero_grad()
        with autocast():
            self.forward()
            
            self.train_loss_l1_1 = self.loss_function_l1(self.restored_images_1, self.sharp_images_1)

            self.train_loss_all =   self.train_loss_l1_1

                            
            if torch.isnan(self.train_loss_all).any():
                raise RuntimeError('NAN!!!')
            if torch.isinf(self.train_loss_all).any():
                raise ZeroDivisionError('INF!!!')
            # self.train_loss_all = self.train_loss_mse_1 + self.train_loss_mse_2 + self.train_loss_mse_3
        
        self.train_loss_all.backward()
        torch.nn.utils.clip_grad_norm_(self.net.convlstm.parameters(),3)
        self.optimizer.step()

        self.update_meters(True, self.blur_images_3.size(0))
        
    def valid_step(self):
        with torch.no_grad():
            with autocast():
                self.forward()
                
                self.valid_loss_l1_1 = self.loss_function_l1(self.restored_images_1, self.sharp_images_1)

                self.valid_loss_all =   self.valid_loss_l1_1
            
            self.update_meters(False, self.blur_images_3.size(0))
        

if __name__ == '__main__':
    ...
