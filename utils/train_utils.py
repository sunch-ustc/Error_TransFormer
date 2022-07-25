import torch
from utils.utils import AverageMeter, ProgressMeter
import pdb
import random
import torch.nn.functional as F
#这个是最初版本的训练函数
 
#这个是memory方法所用的训练函数
def simclr_train(train_loader, model, criterion, optimizer, epoch,image_augmented_num,memory_bank_our,attack="ETF-I"):
 
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    model.train() 
    memory_bank_our.reset()
    
    for i, batch in enumerate(train_loader):
        
        """load data"""
        images = batch['image']
        images_augmented = batch['image_augmented' + str(0)] 
        for i0 in range(image_augmented_num):
            if i0==0:
                continue
            temp=batch['image_augmented'+str(i0)]
            images_augmented = torch.cat([images_augmented,temp]) 
        b, c, h, w = images.size() 
        input_ = torch.cat([images, images_augmented], dim=0)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['label_ind'].cuda(non_blocking=True)
        target=batch['index'].cuda(non_blocking=True)

        """Contrastive learning or Instance classification  """
        if attack=="ETF-C":
            output = model(input_) 
            loss,output0,output1 = criterion(output,memory_bank_our)#,output1  ,output0,output1
            memory_bank_our.update1(output0,output1) 
        elif attack=="ETF-I": 
            output = model(input_)
            tar=torch.cat([targets, targets], dim=0)
            loss=F.cross_entropy(output, tar)
   
        losses.update(loss.item()) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)


 
 
