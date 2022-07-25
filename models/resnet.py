 
import torch.nn as nn
import torchvision.models as models
import pdb #pdb.set_trace()

def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()

    return {'backbone': backbone, 'dim': 2048}

def resnet18():
    backbone = models.__dict__['resnet18']()
    backbone.fc = nn.Identity()
    #print(backbone)
    #pdb.set_trace()
    return {'backbone': backbone, 'dim': 512}
