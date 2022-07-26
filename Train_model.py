from utils.utils import fill_memory_bank,evaluate
import os
import torch
from utils.common_config import get_criterion, get_model, get_train_dataset,get_val_dataset,\
                                  get_train_transformations,\
                                  get_optimizer,get_val_transformations,\
                                adjust_learning_rate
#from offical_simclr import *
from utils.config_test import create_config
from utils.collate import collate_custom
from torch.utils.data import DataLoader
 
from utils.memory import MemoryBank,OurMemory
from utils.train_utils import simclr_train
import numpy as np
import pdb #pdb.set_trace()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Parser
import argparse
from utils.utils import mkdir_if_missing
parser = argparse.ArgumentParser( ) 
 
parser.add_argument("--img_sum", dest='img_sum',  type=int,help="The number of images")
parser.add_argument("--img_num", dest='img_num',  type=int,help="The number of per categary")
parser.add_argument("--batch_size", dest='batch_size',  type=int,help="batch_size")
parser.add_argument("--features_dim", dest='features_dim',  type=int, default=1000, help="features_dim")
parser.add_argument("--backbone", dest='backbone',   type=str,help="backbone")
parser.add_argument("--attack",  type=str,help="The method of training model. option: EFT-I (Instance classification) or EFT-C (Contrastive learning)")
parser.add_argument("--temperature",  type=float,default=0.5)
parser.add_argument("--seed",  type=int,default=0)
parser.add_argument("--epochs",  type=int,default=500)
parser.add_argument("--group",  type=int,default=0) 
parser.add_argument("--lightweight_model", dest='lightweight_model', 
      default="~/results/our_data/checkpoint.pth.tar" ,  type=str,help="lightweight_model")
parser.add_argument("--data_dir", dest='data_dir', 
      default="~/ILSVRC2012_img_val" ,  type=str,help="The path of ILSVRC2012_img_val")
parser.add_argument("--root_path" ,  default="~/" ,type=str, help="The path of root")
args = parser.parse_args() 

def main(): 
 
    """Config""" 
    torch.backends.cudnn.benchmark = True  
    config_exp=os.path.join(args.root_path,'configs/Train_model.yml' )  
    p = create_config( config_exp) 
    for arg in vars(args):
        if getattr(args, arg) is None:
            continue
        print(arg, ':', p[arg])
        p[arg]=getattr(args, arg)
        print(arg, ':', p[arg])
    p["criterion_kwargs"]["temperature"]=p["temperature"]
    
    """Get dataset""" 
    train_transforms = get_train_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms,seed=p["seed"], to_augmented_dataset=True,image_augmented_num=p['image_augmented_num'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                 drop_last=True,shuffle=True)#drop_last=True,#get_train_dataloader(p, dataset)
    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=p['num_workers'],
                batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                 drop_last=False,shuffle=False) 
    memory_bank_our= OurMemory(p["img_sum"],p['features_dim'],p)
    memory_bank_our.cuda()

    """Create model""" 
    model = get_model(p) 
    print(model)
    model=model.cuda() 
    model = torch.nn.DataParallel(model)  
    criterion = get_criterion(p)
    criterion = criterion.cuda()
    optimizer = get_optimizer(p, model) 
    # Checkpoint
    if os.path.exists(p['lightweight_model']):

        checkpoint = torch.load(p['lightweight_model'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        start_epoch = 0
        model = model.cuda()


    torch.autograd.set_detect_anomaly(True)
 
    for epoch in range( start_epoch,p['epochs']):
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('Adjusted learning rate to {:.5f}'.format(param_group['lr']))
            
        """Training"""
        simclr_train(train_dataloader, model, criterion, optimizer, epoch,image_augmented_num=p['image_augmented_num'],
                attack=p["attack"],memory_bank_our=memory_bank_our) 
        
        """Saving"""
        print('Checkpoint ...')
        model.eval()
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['lightweight_model']) 
        
    """Print result"""
    train_acc = evaluate(train_dataloader,model)
    val_acc   = evaluate(val_dataloader,model)
    store  =' train_acc is '+str(train_acc)+'%  '    
    store1 ='val_acc is  '+str( val_acc)+'% '
    print(store)
    print(store1)
    with open('result.txt', 'a') as f:  
        f.writelines(store)
        f.writelines(store1)
        f.writelines('\n') 

if __name__ == '__main__': 
    main()
