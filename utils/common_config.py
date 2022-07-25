 
import csv
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from models.models import ContrastiveModel
from utils.collate import collate_custom
import pdb #pdb.set_trace()
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, p):
        super(SimCLRLoss, self).__init__()
        self.temperature = p['criterion_kwargs']['temperature']
        self.batch = p['batch_size']
        self.image_augmented_num=p['image_augmented_num']
        self.c = 0
 
        self.lamda=p['lamda']# 此处为 一致性损失 和 投影损失之间的平衡参数
    def forward(self, features,memory=0):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR
        """ 
        #b, n, dim = features.size()
        bn, dim = features.size() 
        mask = torch.eye(self.batch, dtype=torch.float32).cuda()
 
        contrast_features=features 
        anchor = features[:self.batch]  
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature  
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True) 
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, self.image_augmented_num+1) 
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(self.batch).view(-1, 1).cuda(), 0) 
        mask = mask * logits_mask 
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1E-16)   
        bn, dim = features.size()
        bn0 = int(bn / (self.image_augmented_num + 1))
        f = features[:bn0]
        f1 = features[bn0:]
        '''assert (bn == ((self.image_augmented_num+1)*self.batch))'''
        '''memory_feature=memory.get_feature()
        memory.update(f1,f)''' 
        loss =- ((mask * log_prob).sum(1) / (mask.sum(1)+1E-16)).mean() 
        # Mean log-likelihood for positive
        return loss,f,f1

class OUR_dataset(Dataset):
    def __init__(self, p,  mode:str, img_num:tuple or int, transform,img_sum:int,data_dir='  /data/linshiqi047/imagenet/val'
            ,seed=0,data_csv_dir='/home/common/sunch/Error_TransFormer/data/selected_data_my.csv'):
        assert mode in ['train', 'val', 'base'], 'WRONG DATASET MODE' 
        super(OUR_dataset).__init__()
        self.mode = mode
        self.group=p["group"]
        self.transform=transform
        self.data_dir = data_dir
        self.seed= seed
        self.img_sum=img_sum        #The number of images
        self.img_num=img_num        #The number of per categary
        label_csv = open('/home/common/sunch/Error_TransFormer/data/imagenet_label.csv', 'r')
        label_reader = csv.reader(label_csv)
        label_ls = list(label_reader)
        self.label_ls = label_ls
        label = {} 
        index=0
        for i in label_ls:          
            label[i[0]] = (i[1:],index)
            index+=1
        self.label=label                # Utilize self.label to record all labels
        data_csv = open(data_csv_dir, 'r')
        csvreader = csv.reader(data_csv)
        data_ls = list(csvreader)
        self.imgs = self.prep_imgs_dir(data_ls) 
        
    def prep_imgs_dir(self, data_ls):
        imgs_ls = []
        
        if self.mode=="val":
            sel_ls_init = list(range(50-self.img_num))
            sel_ls=[i+self.img_num for i in sel_ls_init]
        else:
            sel_ls_init = list(range(self.img_num))
            sel_ls=[i+self.seed*self.img_num for i in sel_ls_init]
        imgs_ls += self.img_ls(data_ls, sel_ls) 
        return imgs_ls
    def img_ls(self, data_ls, sel_ls): 
        imgs_ls = []
        selected_data_csv = open('/home/common/sunch/Error_TransFormer/data/selected_data_my.csv', 'r')
        csvreader = csv.reader(selected_data_csv)
        a=list(csvreader)
        index_num=0
        class_num=0
        n = int(self.img_sum / self.img_num)
        group = self.group
        for label_ind in range(n * group, n * (group + 1)):#len(data_ls)是总的类别数
            target=self.label[a[label_ind][0]][1]
            for img_ind in sel_ls:
                imgs_ls.append([self.data_dir + '/' + a[label_ind][0] + '/' + a[label_ind][1 + img_ind],
                                target,self.label[a[label_ind][0]][0],index_num,class_num])
                index_num+=1
            class_num+=1
        return imgs_ls
 
    def __getitem__(self, item):
        img = Image.open(self.imgs[item][0])
        img_size=img.size
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform['standard'](img)
        dataset={'image': img, 'target': self.imgs[item][1],'index':self.imgs[item][3],'label_ind':self.imgs[item][4],
                 'class_name':self.imgs[item][2],'meta': {'im_size': img_size, 'index': self.imgs[item][1],
                                                          'class_name':self.imgs[item][2] }}
        return dataset
    def __len__(self):
        return len(self.imgs)
def get_criterion(p):
    criterion = SimCLRLoss(p) 
    return criterion
def get_model(p ):
    # Get backbone
    if p['backbone'] == 'resnet18': 
        from models.resnet_vision import resnet18 
        print( p['backbone'])
        backbone0=resnet18()
        backbone0.fc = torch.nn.Identity()
        backbone = {'backbone':backbone0, 'dim': 512}
    elif p['backbone'] == 'resnet10':
        from models.resnet_vision import resnet10
        backbone0=resnet10()
        backbone0.fc = torch.nn.Identity()
        backbone = {'backbone':backbone0, 'dim': 512}

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name'] or 'our_data' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()  

        else:
            raise NotImplementedError 
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone'])) 
    
    
    model = ContrastiveModel(backbone,  head= p['head'], features_dim=p['features_dim'] ) 

 

    return model

def get_train_dataset(p, transform, to_augmented_dataset=False,seed=0,
                        to_neighbors_dataset=False, split=None,image_augmented_num=1):
    # Base dataset
    dataset=OUR_dataset(p,data_dir = p['data_dir'],#'data/ILSVRC2012_img_val',  #存放图片的路径
                          data_csv_dir='/home/common/sunch/Error_TransFormer/data/selected_data_my.csv', #存放索引的路径
                          mode='train',                          #选择生成的模式
                          img_num = p['img_num'],                          #每个类别挑选的图片数
                          transform = transform,
                          seed=seed,
                          img_sum=p['img_sum']                            #一共挑选的图片综述数
)
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset,image_augmented_num=image_augmented_num)
    return dataset


def get_val_dataset(p, transform, seed=0,
                        image_augmented_num=1):
    # Base dataset
    dataset=OUR_dataset(p,data_dir = p['data_dir'],#'data/ILSVRC2012_img_val',  #存放图片的路径
                          data_csv_dir='/home/common/sunch/Error_TransFormer/data/selected_data_my.csv', #存放索引的路径
                          mode='val',                          #选择生成的模式
                          img_num = p['img_num'],                          #每个类别挑选的图片数
                          transform = transform,
                          seed=seed,
                          img_sum=p['img_sum']                            #一共挑选的图片综述数
)

    return dataset

def get_train_transformations(p):
    
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        trans={}
        trans['standard']=transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),  #这一步是水平翻转，竖直翻转可能也会有不错的效果
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
        trans['augment']={}
        trans['augment']['t1'] = transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),  #这一步是水平翻转，竖直翻转可能也会有不错的效果
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

        return trans
 

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    trans={}
    trans['standard']=transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    return trans


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads']) 
    else:
        params = model.parameters()
                
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs']) 
    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs']) 
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 2)+0.004
        #eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 1)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
