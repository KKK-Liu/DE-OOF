import torch.utils.data
import torch

from torch.cuda.amp import autocast as autocast

from .ATT_Deblur_model import ATT_Deblur_Net



class ATT_Deblur_Net_level1(ATT_Deblur_Net):
    def __init__(self, args) -> None:
        super(ATT_Deblur_Net_level1, self).__init__()
        ATT_Deblur_Net.__init__(args, level=4)
        
        if self.isTrain:
            self.loss_names += ['l1_1','l1_2','l1_3', 'l1_4','consistency_confidence','consistency']
            self.meter_init()

            
    def train_step(self):
        self.optimizer.zero_grad()
        with autocast():
            self.forward()
            
            self.train_loss_l1_1 = self.loss_function_l1(self.restored_images_1, self.sharp_images_1)
            self.train_loss_l1_2 = self.loss_function_l1(self.restored_images_2, self.sharp_images_2)
            self.train_loss_l1_3 = self.loss_function_l1(self.restored_images_3, self.sharp_images_3)
            self.train_loss_l1_4 = self.loss_function_l1(self.restored_images_4, self.sharp_images_4)
            
            self.train_loss_consistency_confidence =\
                self.loss_function_l1(self.upsample_fn(self.attention_4, (56 ,56 )), self.attention_3) +\
                self.loss_function_l1(self.upsample_fn(self.attention_3, (112,112)), self.attention_2) +\
                self.loss_function_l1(self.upsample_fn(self.attention_2, (224,224)), self.attention_1)
                
            self.train_loss_consistency = \
                self.loss_function_l1(self.upsample_fn(self.restored_images_4, (56 ,56 )), self.restored_images_3) +\
                self.loss_function_l1(self.upsample_fn(self.restored_images_3, (112,112)), self.restored_images_2) +\
                self.loss_function_l1(self.upsample_fn(self.restored_images_2, (224,224)), self.restored_images_1)
            
            self.train_loss_all =   self.train_loss_l1_1 + self.train_loss_l1_2 + self.train_loss_l1_3 +\
                                    self.lambda_RR * self.train_loss_consistency/3 +\
                                    self.lambda_CM * self.train_loss_consistency_confidence/3
                            
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
                self.valid_loss_l1_2 = self.loss_function_l1(self.restored_images_2, self.sharp_images_2)
                self.valid_loss_l1_3 = self.loss_function_l1(self.restored_images_3, self.sharp_images_3)
                self.valid_loss_l1_4 = self.loss_function_l1(self.restored_images_4, self.sharp_images_4)
                
                self.valid_loss_consistency_confidence =\
                    self.loss_function_l1(self.upsample_fn(self.attention_4, (56 ,56 )), self.attention_3) +\
                    self.loss_function_l1(self.upsample_fn(self.attention_3, (112,112)), self.attention_2) +\
                    self.loss_function_l1(self.upsample_fn(self.attention_2, (224,224)), self.attention_1)
                    
                self.valid_loss_consistency = \
                    self.loss_function_l1(self.upsample_fn(self.restored_images_4, (56 ,56 )), self.restored_images_3) +\
                    self.loss_function_l1(self.upsample_fn(self.restored_images_3, (112,112)), self.restored_images_2) +\
                    self.loss_function_l1(self.upsample_fn(self.restored_images_2, (224,224)), self.restored_images_1)
                
                self.valid_loss_all =   self.valid_loss_l1_1 + self.valid_loss_l1_2 + self.valid_loss_l1_3 +\
                                        self.lambda_RR * self.valid_loss_consistency/3 +\
                                        self.lambda_CM * self.valid_loss_consistency_confidence/3
            
            self.update_meters(False, self.blur_images_3.size(0))
        

if __name__ == '__main__':
    ...
