import torchvision.models as models

#from pl_bolts.models.self_supervised import SimCLR
import argparse
 
import sys
import torch
import torchvision.transforms as transforms
from utils.utils import fill_memory_bank
import matplotlib.pyplot as plt
from utils.memory import MemoryBank,OurMemory
from attack.model import *
from attack.low_level import *
from utils.utils import AverageMeter
from utils.collate import collate_custom
from utils.config_test import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_train_transformations,\
                                get_optimizer,\
                                adjust_learning_rate
import pdb #pdb.set_trace()
import os
from test import test_function
from PIL import ImageFile
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
def main(p,path_save_adv_image='/home/common/sunch/Error_TransFormer/image_adv',seed=0,layer=["layer1"], 
            path_feature_extractor="/home/common/sunch/Error_TransFormer/results/resnet/resnet18_1w.pth.tar"  ):
    
    torch.backends.cudnn.benchmark = True 
    print(path_feature_extractor)
    print(p["backbone"])
    print(p["attack_method"])
    attack_method=p["attack_method"]
    trans={}
    trans['standard']=transforms.Compose([ transforms.CenterCrop(p['transformation_kwargs']['crop_size']), transforms.ToTensor()]) 
    trans['augment']={'t1': trans['standard']} 
    
    criterion = get_criterion(p)
    criterion = criterion.cuda()
    
    """Get dataset""" 
    dataset = get_train_dataset(p, trans,seed=seed, to_augmented_dataset=False,image_augmented_num=p['image_augmented_num'])
    train_dataset = get_train_dataset(p, trans,seed=seed, to_augmented_dataset = True,image_augmented_num=p['image_augmented_num'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers = p['num_workers'],
                batch_size = p['batch_size'], pin_memory = True, collate_fn = collate_custom,
                drop_last = True,shuffle=False)#drop_last=True,#get_train_dataloader(p, dataset)
    memory_bank_our= OurMemory(len(train_dataset),
                                    p['features_dim'],p)
    memory_bank_our.cuda()
    
    
    """Load model""" 
    model = get_model(p)  
    myexactor = FeatureExtractor(model.backbone ,["layer1"]) 
    model.cuda()
    model = torch.nn.DataParallel(model) 
    myexactor.cuda()
    myexactor = torch.nn.DataParallel(myexactor)  
    checkpoint = torch.load(path_feature_extractor, map_location='cpu') 
    model.load_state_dict(checkpoint['model'])
    
    for para in model.parameters():
        para.requires_grad = False
    model.eval()
    myexactor.eval() 
    tp=0

    
    for batch in train_dataloader: 
        
        images = batch['image'].cuda(non_blocking=True) 
        target = batch['index'].cuda(non_blocking=True)  
        order=np.arange(images.shape[0])[::-1]
        guide_image=images[order.copy()].detach()   
        target=target[order.copy()].detach() 
        
        
        """Mount attacks"""
        if  attack_method == "Deep_PGD": 
            adv_images = Deep_PGD(model, images , target) 
        elif attack_method == "ETF_PGD":
            adv_images = ETF_PGD(myexactor, images ,guide_image,eps=0.1,  iters=200  )  
        
   
        """Save images"""
        save_dir = path_save_adv_image
        os.makedirs(save_dir , exist_ok=True) 
        batch_size = p['batch_size']
        for save_ind in range(batch_size):
            file_path, file_name = dataset.imgs[tp * batch_size + save_ind][0].split('/')[-2:]
            os.makedirs(save_dir + '/' + file_path, exist_ok=True) 
            save_attack_img(img = adv_images[save_ind],file_dir = os.path.join(save_dir, file_path, file_name[:-5]) + '.PNG') 
            if (tp * batch_size + save_ind)%10 == 0:
                print('\r', tp * batch_size + save_ind, 'images saved.', end=' ') 
        tp += 1


parser = argparse.ArgumentParser( )
if __name__ == '__main__':
    parser.add_argument("--path_save_adv_image", default="/home/common/sunch/Error_TransFormer/images/image_adv" ,  type=str)
    parser.add_argument("--path_feature_extractor",default="/home/common/sunch/Error_TransFormer/results/ETF-I/I-resnet10_1k_seed1.pth.tar" ,  type=str)
    parser.add_argument("--backbone",             default="resnet10" ,  type=str)
    parser.add_argument("--attack",             default="ETF-I" ,  type=str)
    parser.add_argument("--features_dim",     default=1000 ,  type=int)
    parser.add_argument("--attack_all",       default=["PGD","TI","DI","MI"]  )
    parser.add_argument("--seed",       default=0,  type=int)
    parser.add_argument("--group",       default=0,  type=int)
    parser.add_argument("--temperature",       default=0.1,  type=float)
    parser.add_argument("--img_sum",       default=1000,  type=int)
    parser.add_argument("--img_num",       default=1,  type=int)
    parser.add_argument("--batch_size",       default=25,  type=int)
    parser.add_argument("--beta",       default=1,  type=int)
    parser.add_argument("--attack_method",       default="ETF_PGD",  type=str )
    parser.add_argument("--log_path",       default='./result.txt',  type=str )
    parser.add_argument("--data_dir", dest='data_dir', 
      default="/home/common/sunch/ILSVRC2012_img_val" ,  type=str,help="The path of ILSVRC2012_img_val")
    args = parser.parse_args()
    print(args)
    
    config_env='/home/common/sunch/Error_TransFormer/configs/env.yml' 
    config_exp='/home/common/sunch/Error_TransFormer/configs/test/ETF_Attack.yml'
    p = create_config(config_env,config_exp)
    
    """ """
    path_save_adv_image   =       args.path_save_adv_image  
    p["backbone"]         =       args.backbone  
    path_feature_extractor=       args.path_feature_extractor  
    p["attack"]           =       args.attack
    p["features_dim"]     =       args.features_dim
    p["criterion_kwargs"]["temperature"]=args.temperature 
    p["img_sum"]          =       args.img_sum
    p["img_num"]          =       args.img_num
    p["batch_size"]       =       args.batch_size
    p["seed"]             =       args.seed
    p["group"]            =       args.group
    p["beta"]            =       args.beta 
    p["attack_method"]   =       args.attack_method 
    p["data_dir"]         =       args.data_dir 
    log_path             =       args.log_path
     
 
    main(p,path_feature_extractor=path_feature_extractor,seed=p["seed"],path_save_adv_image=path_save_adv_image  ) 

     
    test_function(c1=['vgg19_bn','inception_v3','resnet152', 'densenet161','squeezenet1_0','WRN','mobilenet_v2'],#'resnet50',"adv_trained4","adv_trained"  ,"adv_trained"  'vgg19_bn','inception_v3','resnet152', 'densenet161','squeezenet1_0','WRN','mobilenet_v2'],
        path_save_adv_image=path_save_adv_image,group=p["group"],seed=p["seed"],img_sum=p["img_sum"],img_num=p["img_num"],log_path=log_path  )
