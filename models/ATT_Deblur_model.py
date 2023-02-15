import torch.nn as nn
import torch.utils.data
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ConstantLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from .base_model import BaseModel
from torch.cuda.amp import autocast as autocast

from torch.nn.init import xavier_normal_ , kaiming_normal_
from functools import partial

import torchvision.transforms.functional as f
import os
import numpy as np

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, input_chans, num_features, filter_size ):
        super(CLSTM_cell, self).__init__()
        
        #self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        #self.batch_size=batch_size
        self.padding=(filter_size-1)//2#in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)

    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state#hidden and c are images with several channels
        #print 'hidden ',hidden.size()
        #print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)#oncatenate in the channels
        #print 'combined',combined.size()
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.num_features,dim=1)#it should return 4 tensors
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
        
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self,batch_size,shape):
        return (torch.zeros(batch_size,self.num_features,shape[0],shape[1]).cuda() , torch.zeros(batch_size,self.num_features,shape[0],shape[1]).cuda())
        # return (torch.zeros(batch_size,self.num_features,shape[0],shape[1]) , torch.zeros(batch_size,self.num_features,shape[0],shape[1]))

def get_weight_init_fn(activation_fn):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr( activation_fn , 'func' ):
        fn = activation_fn.func

    if  fn == nn.LeakyReLU:
        negative_slope = 0 
        if hasattr( activation_fn , 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr( activation_fn , 'args'):
            if len( activation_fn.args) > 0 :
                negative_slope = activation_fn.args[0]
        return partial( kaiming_normal_ ,  a = negative_slope )
    elif fn == nn.ReLU or fn == nn.PReLU :
        return partial( kaiming_normal_ , a = 0 )
    else:
        return xavier_normal_
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation_fn= None , use_batchnorm = False , pre_activation = False , bias = True , weight_init_fn = None ):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    try:
        weight_init_fn( conv.weight )
    except:
        print( conv.weight )
    layers.append( conv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation_fn = None ,   use_batchnorm = False , pre_activation = False , bias= True , weight_init_fn = None ):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( deconv.weight )
    layers.append( deconv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )


class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True ) , pre_activation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , use_batchnorm )
        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , None , use_batchnorm  , weight_init_fn = get_weight_init_fn(last_activation_fn) )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm )
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor
    def forward(self , x ):
        residual = x 
        if self.downsample is not None:
            residual = self.downsample( residual )

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out


def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False, activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(1):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5_relu(
            in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(1):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, out_channels, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.convlstm = CLSTM_cell(128, 128, 5)
        
        self.dblock1_content = DBlock(128, 64, 2, 1)
        self.dblock2_content = DBlock(64, 32, 2, 1)
        self.outblock_content = OutBlock(32, 3)
        
        self.dblock1_attention = DBlock(128, 64, 2, 1)
        self.dblock2_attention = DBlock(64, 32, 2, 1)
        self.outblock_attention = OutBlock(32, 3)

        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)

    def forward_step(self, x, hidden_state):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h, c = self.convlstm(e128, hidden_state)
        
        d64_content = self.dblock1_content(h)
        d32_content = self.dblock2_content(d64_content + e64)
        d3_content = self.outblock_content(d32_content + e32)
        
        d64_attention = self.dblock1_attention(h)
        d32_attention = self.dblock2_attention(d64_attention + e64)
        d3_attention = self.outblock_attention(d32_attention + e32)
        
        d3_content = torch.tanh(d3_content)
        d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        
        xs = list(torch.split(x, 3, 1))
        
        d3 = d3_content * d3_attention + (xs[0]+xs[1]) * (1 - d3_attention)

        return d3, h, c, d3_attention

    def forward(self, b1, b2, b3):
        h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2]//4, b3.shape[-1]//4))

        i3, h, c, a3 = self.forward_step(
            torch.cat([b3, b3], 1), (h, c))

        c = self.upsample_fn(c, scale_factor=2)
        h = self.upsample_fn(h, scale_factor=2)
        i2, h, c, a2 = self.forward_step(
            torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

        c = self.upsample_fn(c, scale_factor=2)
        h = self.upsample_fn(h, scale_factor=2)
        i1, h, c, a1 = self.forward_step(
            torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))

        return i1, i2, i3, a1, a2, a3

class ATT_Deblur_Net(nn.Module, BaseModel):
    def __init__(self, args) -> None:
        super(ATT_Deblur_Net, self).__init__()
        BaseModel.__init__(self, args)
        
        self.net = Net()
        self.nets.append(self.net)
        self.isTrain = self.args.isTrain
        if self.isTrain:
            if args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam( 
                    self.net.parameters(),
                    lr = args.lr,
                    weight_decay = args.weight_decay
                )
            elif args.optimiazae == 'sgd':
                self.optimizer =optim.SGD(
                    self.net.parameters(), 
                    lr=args.lr,
                    momentum=args.momentum, 
                    nesterov=True, 
                    weight_decay=args.weight_decay)
            
            if args.scheduler == 'multisteplr':
                self.scheduler = MultiStepLR(self.optimizer, args.milestones, args.gamma)
            elif args.scheduler == 'cosineannealinglr':
                self.scheduler = CosineAnnealingLR(self.optimizer,T_max=args.T_max)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
            
            self.schedulers.append(self.scheduler)
            self.loss_function_mse = nn.MSELoss()
            self.loss_function_l1 = nn.L1Loss()
            # self.loss_function_ssim = 
            # self.loss_names += ['mse_1','mse_2','mse_3']
            # self.train_loss_names += ['l1_1','l1_2','l1_3','consistency_confidence','consistency']
            # self.valid_loss_names += ['l1_1','l1_2','l1_3','consistency_confidence','consistency']
            self.loss_names += ['l1_1','l1_2','l1_3','consistency_confidence','consistency']
            self.meter_init()
            self.upsample_fn = partial(torch.nn.functional.interpolate, mode='bilinear')
        else:
            self.result_save_root = args.result_save_root 
            self.visual_names = ['restored_images_1']
            self.eval_losses = []
            self.image_names = []

        print('SRNATTS_Net is created')
        
    def get_visuals(self):
        with torch.no_grad():
            with autocast():
                self.forward()
                
    def save_visuals(self):
        for visual_name in self.visual_names:
            visual_batch = getattr(self,visual_name).detach().cpu()
            for image, name in zip(visual_batch, self.paths):
                image = f.to_pil_image(image)
                save_file_name = os.path.join(
                    self.result_save_root,
                    'image',
                    '{}_{}.png'.format(name.replace('.png',''),visual_name)
                )
                image.save(save_file_name)
        
        
    def eval_visuals(self, metrics:dict):
        self.image_names += self.paths
        
        for image_restored, image_sharp in zip(self.restored_images_1, self.sharp_images_1):
            this_losses = []
            for _, metric_function in metrics.items():
                this_losses.append(metric_function(image_restored.unsqueeze(0), image_sharp.unsqueeze(0)).data)
            self.eval_losses.append(this_losses)

    def eval_result_save(self, metrics:dict):
        with open(os.path.join(self.result_save_root, 'metric values.csv'), 'w') as f:
            line = ','.join(['image name']+list(metrics.keys()))+'\n'
            f.write(line)
            for this_losses, image_name in zip(self.eval_losses, self.image_names):
                line = ','.join([image_name]+list(map(str, this_losses))) + '\n'
                f.write(line)


        self.eval_losses = np.array(self.eval_losses)
        np.save(os.path.join(self.result_save_root,'eval_losses.npy'), self.eval_losses)
        
        losses_item = np.mean(self.eval_losses, axis=0)

        with open(os.path.join(self.result_save_root, 'eval_result.txt'), 'w') as f:
            for metric, value in zip(metrics.keys(), losses_item):
                f.write("{:>20}:{:<20}\n".format(metric, value))
                
        with open(os.path.join(self.result_save_root, 'eval_result.txt'), 'r') as f:
            for line in f.readlines():
                print(line)
            
    def to_cuda(self):
        self.net = self.net.cuda()
        self.loss_function_mse = self.loss_function_mse.cuda()
        self.loss_function_l1 = self.loss_function_l1.cuda()
        return self
        
    def set_input(self, data):
        # self.blur_images_3, self.blur_images_2, self.blur_images_1,\
        # self.sharp_images_3, self.sharp_images_2, self.sharp_images_1 = data
        if self.isTrain:
            self.blur_images_1, self.blur_images_2, self.blur_images_3,\
                self.sharp_images_1, self.sharp_images_2, self.sharp_images_3 = data
        else:
            self.blur_images_1, self.blur_images_2, self.blur_images_3,\
                self.sharp_images_1, self.sharp_images_2, self.sharp_images_3, self.paths = data
        self.blur_images_3  = self.blur_images_3.cuda(non_blocking = True)
        self.blur_images_2  = self.blur_images_2.cuda(non_blocking = True)
        self.blur_images_1  = self.blur_images_1.cuda(non_blocking = True)
        
        self.sharp_images_1 = self.sharp_images_1.cuda(non_blocking = True)
        self.sharp_images_2 = self.sharp_images_2.cuda(non_blocking = True)
        self.sharp_images_3 = self.sharp_images_3.cuda(non_blocking = True)
        
    def forward(self):
        self.restored_images_1, self.restored_images_2, self.restored_images_3,\
        self.attention_1, self.attention_2, self.attention_3 = self.net(self.blur_images_1, self.blur_images_2, self.blur_images_3)
        
        # self.attention_1 = self.upsample_fn(self.attention_1, size=(224,224))
        # self.attention_2 = self.upsample_fn(self.attention_2, size=(224,224))
        
        
    def train_step(self):
        self.optimizer.zero_grad()
        with autocast():
            self.forward()
            
            self.train_loss_l1_1 = self.loss_function_l1(self.restored_images_1, self.sharp_images_1)
            self.train_loss_l1_2 = self.loss_function_l1(self.restored_images_2, self.sharp_images_2)
            self.train_loss_l1_3 = self.loss_function_l1(self.restored_images_3, self.sharp_images_3)
            
            self.train_loss_consistency_confidence =\
                self.loss_function_l1(self.upsample_fn(self.attention_3, (112,112)), self.attention_2) +\
                self.loss_function_l1(self.upsample_fn(self.attention_2, (224,224)), self.attention_1)
                
            self.train_loss_consistency = \
                self.loss_function_l1(self.upsample_fn(self.restored_images_3, (112,112)), self.restored_images_2) +\
                self.loss_function_l1(self.upsample_fn(self.restored_images_2, (224,224)), self.restored_images_1)
            
            self.train_loss_all =   self.train_loss_l1_1 + self.train_loss_l1_2 + self.train_loss_l1_3 +\
                                    self.train_loss_consistency/2 + self.train_loss_consistency_confidence/2
                            
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
                
                self.valid_loss_consistency_confidence =\
                    self.loss_function_l1(self.upsample_fn(self.attention_3, (112,112)), self.attention_2) +\
                    self.loss_function_l1(self.upsample_fn(self.attention_2, (224,224)), self.attention_1)
                    
                self.valid_loss_consistency = \
                    self.loss_function_l1(self.upsample_fn(self.restored_images_3, (112,112)), self.restored_images_2) +\
                    self.loss_function_l1(self.upsample_fn(self.restored_images_2, (224,224)), self.restored_images_1)
                
                self.valid_loss_all =   self.valid_loss_l1_1 + self.valid_loss_l1_2 + self.valid_loss_l1_3 +\
                                        self.valid_loss_consistency/2 + self.valid_loss_consistency_confidence/2
            
            # self.valid_loss_mse_1 = self.loss_function_mse(self.restored_images_1, self.sharp_images_1)
            # self.valid_loss_mse_2 = self.loss_function_mse(self.restored_images_2, self.sharp_images_2)
            # self.valid_loss_mse_3 = self.loss_function_mse(self.restored_images_3, self.sharp_images_3)
            # self.valid_loss_all = self.valid_loss_mse_1 + self.valid_loss_mse_2 + self.valid_loss_mse_3
            
            self.update_meters(False, self.blur_images_3.size(0))
        
        
    def load_network(self):
        ckpt = torch.load(self.args.load_ckpt_path)
        if self.isTrain:
            self.net.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
        else:
            self.net.load_state_dict(ckpt['state_dict'])
            
    def get_and_save_visual_results(self):
        with torch.no_grad():
            with autocast():
                self.forward()
                
                for img_tensor, name in zip(self.restored_images_1, self.paths):
                    img_rgb = f.to_pil_image(img_tensor)
                    img_rgb.save(os.path.join(self.result_save_root, name))
        
if __name__ == '__main__':
    ...
