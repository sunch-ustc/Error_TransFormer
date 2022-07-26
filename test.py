from random import seed
import torchvision.models as models

#from pl_bolts.models.self_supervised import SimCLR

#from offical_simclr import *

import sys
import torch
import torchvision.transforms as transforms
from utils.utils import fill_memory_bank
import matplotlib.pyplot as plt
from utils.memory import MemoryBank,OurMemory
from attack.model import *
from utils.utils import AverageMeter
from utils.collate import collate_custom
from utils.config_test import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                 get_train_transformations,\
                                 get_optimizer,\
                                adjust_learning_rate
import pdb #pdb.set_trace()
import os
from torch.utils.data import Dataset
import csv
 
torch.backends.cudnn.benchmark = True
#torch.cuda.set_device(1)

class test_dataset(Dataset):
    def __init__(self, p0,  mode:str, img_num:tuple or int, transform,img_sum:int,group=0,data_dir='~/Error_TransFormer_bithub/image_adv',
                  seed=0, data_csv_dir='~/Error_TransFormer_bithub/data/selected_data_my.csv'):
        assert mode in ['train', 'val', 'base'], 'WRONG DATASET MODE'
        #assert img_num in [1,5,10,20], 'ONLY SUPPORT 2/10/20/40 IMAGES'
        super(test_dataset).__init__()
        self.p0=p0
        self.mode = mode
        self.transform=transform
        self.group=group
        self.data_dir = data_dir
        self.seed=seed
        self.img_sum=img_sum        #一共挑取多少张图片
        self.img_num=img_num        #每个类别挑选的图像数量
        label_csv = open(os.path.join(self.p0['root_path'],'data/imagenet_label.csv'), 'r')
        label_reader = csv.reader(label_csv)
        label_ls = list(label_reader)
        self.label_ls = label_ls
        label = {}#EasyDict()
        index=0
        for i in label_ls:  # 将标签都记录在self.label中
            label[i[0]] = (i[1:],index)
            index+=1
        self.label=label
        data_csv = open( os.path.join(self.p0['root_path'],'data/selected_data.csv') , 'r')
        csvreader = csv.reader(data_csv)
        data_ls = list(csvreader)
        self.imgs = self.prep_imgs_dir(data_ls)
    def prep_imgs_dir(self, data_ls):
        imgs_ls = []
        sel_ls_init = list(range(self.img_num))
        sel_ls=[i+self.seed*self.img_num for i in sel_ls_init]
        imgs_ls += self.img_ls(data_ls, sel_ls)#  sel_ls 是一个序列 比如最初给n_imgs付20  实际n_imgs为10   sel_ls就是 0 1 2 3 ... 9
        return imgs_ls
    def img_ls(self, data_ls, sel_ls):# 该函数将图片一张张储存到imgs_ls中
        imgs_ls = []
        selected_data_csv = open(os.path.join(self.p0['root_path'],'data/csv/test_data.csv'), 'r')
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
        dataset={'image': img, 'target': self.imgs[item][1],'index':self.imgs[item][3],
                 'class_name':self.imgs[item][2],'meta': {'im_size': img_size, 'index': self.imgs[item][1],'label_ind':self.imgs[item][4],
                                                          'class_name':self.imgs[item][2] }}
        return dataset
    def __len__(self):
        return len(self.imgs)

def test_function(p0,path_save_adv_image='~/Error_TransFormer_bithub/image_adv',seed=0,group=0,
    c1=[  'vgg19_bn','inception_v3','resnet152', 'densenet161','squeezenet1_0','WRN','mobilenet_v2'],img_num=1,img_sum=1000,log_path='result.txt' ):
    
    config_exp=os.path.join(p0['root_path'],'/configs/test.yml')
    p = create_config( config_exp)
    p["data_dir"]=path_save_adv_image
    p['img_num']= img_num
    p['img_sum']= img_sum

    if p['img_sum']<20:
        p['batch_size']=img_sum
    print(path_save_adv_image)
    #train_transforms = get_train_transformations(p)
    trans={}
    trans['standard']=transforms.Compose([
                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                transforms.ToTensor()])
                #transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    trans['augment']={}
    trans['augment']['t1']=transforms.Compose([
                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                transforms.ToTensor()])
                #transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    train_dataset = test_dataset(
                          p0=p0,                          
                          data_dir = p['data_dir'],#'data/ILSVRC2012_img_val',  #存放图片的路径
                          mode='train',                          #选择生成的模式
                          img_num = p['img_num'],                          #每个类别挑选的图片数
                          transform = trans,
                          seed=seed,
                          img_sum=p['img_sum'] ,
                          group=group                           #一共挑选的图片综述数
)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                drop_last=True,shuffle=False)#drop_last=True,#get_train_dataloader(p, dataset)
    memory_bank_our= OurMemory(len(train_dataset),
                                    p['model_kwargs']['features_dim'],p)
    memory_bank_our.cuda()
    top2 = AverageMeter('Acc@1', ':6.2f')
    exact_list=["layer2"]
 
 
     
    acc = {} 
    #modl=FeatureExtractor(model0,exact_list)
    for i in c1 :
        model = defend_model(i)
        model.eval()
        #print(model)
        model.cuda()
        #model = torch.nn.DataParallel(model)
        top1 = AverageMeter('Acc@1', ':6.2f')
        tp=0
        
        for batch in train_dataloader:
            #获取对抗样本

            images = batch['image'].cuda(non_blocking=True)
            #modl(images)
            target = batch['target'].cuda(non_blocking=True)
            index=batch['index'].cuda(non_blocking=True)

            
            output=model(norm_layer(images)) if i!= "adv_trained" else model(images)
            
            #pdb.set_trace()
            predict = torch.topk(output, 1)[1].squeeze(1)
            acc1 = 100*torch.mean(torch.eq(predict, target).float())

            top1.update(acc1.item(), images.size(0))
            tp += 1
            if tp%50==49:

                #print('Result of ' + i + ' evaluation is %.2f' % (top1.avg) + '% in '+str(tp)+' batch')
                acc[i]=top1.avg
                acc['avg']=top2.avg
                
        
        store='Result of ' + i + ' evaluation is %.2f' % (top1.avg) + '%'
        print(store)
        with open(log_path, 'a') as f: # 写文件, 以行的方式写, 传列表格式
            f.writelines(store)
            f.writelines('\n')
            ''''''
        top2.update(top1.avg, 1)
        acc[i] = top1.avg
        acc['avg'] = top2.avg
        
        torch.cuda.empty_cache()
    store='Result of  evaluation is %.2f' % (top2.avg)+'%'
    print(store)

    with open(log_path, 'a') as f: # 写文件, 以行的方式写, 传列表格式
        f.writelines(store)
        f.writelines('\n')
        f.writelines('----------------------------------------------  \n')
if __name__ == '__main__':

    test_function(path_save_adv_image='~/Error_TransFormer_bithub/images/image_adv',seed=1,
    c1=['vgg19_bn','inception_v3','resnet152', 'densenet161','squeezenet1_0','WRN','mobilenet_v2'],img_num=1,img_sum=1000)
