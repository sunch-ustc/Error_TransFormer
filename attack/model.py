import torchvision.models as models
import torch
import PIL
import torch.nn.functional as F
from PIL import Image
#from skimage import io
import scipy.stats as st
import numpy as np 
import pdb #pdb.set_trace()
 
import torchvision.transforms as T
import numpy as np
from robustness import model_utils 
from robustness.datasets import ImageNet
 
import torch
 
 
 
def norm_layer(images):
    mean=torch.tensor((0.485, 0.456, 0.406))
    std=torch.tensor((0.229, 0.224, 0.225))
    dtype = images.dtype
    images=images.clone()#.detach()
    mean = torch.as_tensor(mean, dtype=dtype, device=images.device)
    std = torch.as_tensor(std, dtype=dtype, device=images.device)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    images.sub_(mean).div_(std)
    return images
class Advtrain(torch.nn.Module):

    def __init__(self, backbone):
            super(Advtrain, self).__init__()
            #self.backbone = backbone
            # self.pool=nn.AvgPool2d()
            #self.module = torch.nn.Sequential(backbone)
            self.add_module("module", backbone)
    def forward(self, x):
            features = x  # F.interpolate(x, size=[96, 96])

            features = self.module(features)
            features=features[0]
            return features
def defend_model(string):
    if string=='adv_trained':
        ds = ImageNet(' /data/linshiqi047/imagenet/val')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,pytorch_pretrained=True)  #
        #print(torch.load('imagenet_linf_4.pt').keys())
        model=Advtrain(model)
        #print(model)
        model.load_state_dict(torch.load('/home/common/sunch/unsupervised-1/imagenet_linf_8.pt')['model'])
    elif string=='adv_trained4':
        ds = ImageNet(' /data/linshiqi047/imagenet/val')
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,pytorch_pretrained=True)  #
        #print(torch.load('imagenet_linf_4.pt').keys())
        model=Advtrain(model) 
        model.load_state_dict(torch.load('/home/common/sunch/unsupervised-1/imagenet_linf_4.pt')['model'])
        
    if string=='vgg19_bn':
        model=models.vgg19_bn(pretrained=True)  
    elif string=='inception_v3':
        model=models.inception_v3(pretrained=True) 
    elif string=='resnet152':
        model=models.resnet152(pretrained=True) 
    elif string=='squeezenet1_0':
        model=models.squeezenet1_0(pretrained=True) 
    elif string=='mobilenet_v2':
        model=models.mobilenet_v2(pretrained=True)  
    elif string=='densenet161':
        model=models.densenet161(pretrained=True) 
    elif string == 'WRN':
        model = models.wide_resnet50_2(pretrained=True)  
    return model

def save_attack_img(img, file_dir):
    T.ToPILImage()(img.data.cpu()).save(file_dir)

 
# 中间特征提取
class FeatureExtractor(torch.nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        #outputs = []
        outputs=0

        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            #pdb.set_trace()
            #print(name)
            #print(outputs)
            if name in self.extracted_layers:
                #outputs.append(x)
              
                outputs=x
                break
        return outputs
class Mid_to_Mid(torch.nn.Module):
    def __init__(self, submodule,input_layer, output_layer ):
        super(Mid_to_Mid, self).__init__()
        self.submodule = submodule
        self.input_layer = input_layer
        self.output_layer = output_layer 
    def forward(self, x):
        #outputs = []
        outputs=0
        con=False
        for name, module in self.submodule._modules.items():
            #if name is "fc": x = x.view(x.size(0), -1)
            
            if name in self.input_layer or con:
                con=True
                if name in self.input_layer: continue
                x = module(x)
                
            if name in self.output_layer: 
                outputs=x
                break
        return outputs
 