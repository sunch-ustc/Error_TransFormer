 
import torchvision.models as models
import time
import sys
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
 
import scipy.stats as st
sys.path.append('../') 
from attack.model import norm_layer 
 
import pdb #pdb.set_trace() 

def augment_layer(image ): 
    '''
    input:    image             (bitchsize, channal, w , h  ) 
    
    output:   image_augment     (bitchsize, augment_num, channal, w , h  ) 
    '''
    b, c, h, w = image.size()
    #image1  
    image1=TF.adjust_brightness(image,1.5) 
    image1=TF.adjust_hue(image1,0.3)   #[-0.5,0.5]
    image1=image1.unsqueeze(1)
    #image2  
    image2=TF.hflip(image)
    image2=TF.adjust_contrast(image2,0.5) 
    image2=image2.unsqueeze(1)
    #image3 # i, j, h, w, self.size, self.interpolation)
    image3=TF.resized_crop(image, int(h/4), int(w/4), int(h/2), int(w/2),  (224,224) )
    image3=TF.adjust_brightness(image3,0.6)
    image3=image3.unsqueeze(1)
    #image4
    image4=TF.rgb_to_grayscale(image, num_output_channels=3)
    image4=TF.adjust_saturation(image4,2)
    image4=image4.unsqueeze(1)
    
    image=image.unsqueeze(1)
    image_augment= torch.cat([image,image1,image2,image3,image4],dim=1)#
    image_augment=image_augment.view(-1, c, h, w)
    return image_augment
def criterion(ori_mid, tar_mid, att_mid):
    bs = ori_mid.shape[0]
    ori_mid = ori_mid.view(bs, -1)
    tar_mid = tar_mid.view(bs, -1)
    att_mid = att_mid.view(bs, -1)
    pert = att_mid - ori_mid
    pert_target = tar_mid - ori_mid    
    pert_target = pert_target  
    loss = (pert * pert_target).sum() / bs
    return loss 
    
def Deep_PGD(model, images , target,tp=0, eps=0.1, alpha=1 / 255, iters=200 ):
      
    orig_images = images.clone()
    img_x = images.data
    img_x.requires_grad = True
    orig_images.requires_grad = False
 
 
    for i in range(iters):
        img_x = images.clone()
        img_x = images.data + images.data.new(images.size()).uniform_(-eps, eps)
        img_x.requires_grad = True
        f = model(norm_layer(img_x))
        loss = - F.cross_entropy(f, target) 
        loss.backward()
        images = images.data + alpha * img_x.grad.sign()

        images = torch.where(images > (orig_images + eps), orig_images + eps, images)
        images = torch.where(images < (orig_images - eps), orig_images - eps, images)
        images = torch.clamp(images, min=0, max=1)
        img_x.grad.data = torch.zeros(img_x.shape).cuda()
    return images
def ETF_PGD(extractor, images, guide_image, eps=0.1, alpha=1 / 255, iters=200, 
                  l_norm =0,   rho=0.0001,skip=20,ratio=0.1):
    # basic settings
     
    # 0: l2-norm; 1: linf-norm

    # initialization
    #  for crafting adv images
    watermark = torch.zeros_like(images).cuda()

    #  for SAM
    s = images.clone()
    t = guide_image.clone()
    a = images.clone()
    rho_s = rho*ratio
    rho_t = rho*ratio
    rho_a = rho
    
    # the main loop
    for i in range(int(iters)):
        pert_s = torch.zeros_like(images).cuda()
        pert_t = torch.zeros_like(images).cuda()
        pert_a = torch.zeros_like(images).cuda()

        # the first step
        if i % skip == 0:   
            # initialization
            pert_s.requires_grad_()
            pert_t.requires_grad_()
            pert_a.requires_grad_() 
            
            # calculate loss
            sou_h_feats = extractor(norm_layer(augment_layer(s + pert_s)))
            tar_h_feats = extractor(norm_layer(augment_layer(t + pert_t)))
            adv_h_feats = extractor(norm_layer(augment_layer(a + pert_a)))
            
            loss= criterion(sou_h_feats, tar_h_feats, adv_h_feats)
            print("loss1:  "+ str(loss))
            # craft perturbation in SAM
            loss.backward()
            if l_norm == 0:
                grad_s = pert_s.grad 
                pert_s = rho_s * grad_s / (torch.sum(grad_s ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16)

                grad_t = pert_t.grad
                pert_t = rho_t * grad_t / (torch.sum(grad_t ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16)

                grad_a = pert_a.grad
                pert_a = rho_a * grad_a / (torch.sum(grad_a ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16)
            else:
                pert_s = torch.sign(pert_s.grad)
                pert_t = torch.sign(pert_t.grad)
                pert_a = torch.sign(pert_a.grad)

        
        # the second step
        watermark.requires_grad_()
        if watermark.grad is not None:
            watermark.grad.data.fill_(0)
        extractor.zero_grad()

        # calculate loss
        sou_h_feats = extractor(norm_layer(augment_layer(s - pert_s))).detach()
        tar_h_feats = extractor(norm_layer(augment_layer(t - pert_t))).detach()
        adv_h_feats = extractor(norm_layer(augment_layer(s - pert_a + watermark)))
        
        loss= criterion(sou_h_feats, tar_h_feats, adv_h_feats)
        
        loss.backward()
        print("loss2:  "+ str(loss))
        # update and clip
        grad = torch.sign(watermark.grad)
        watermark = watermark.detach() + alpha * grad
        watermark = torch.where(watermark > (+ eps), torch.zeros_like(watermark).cuda() + eps, watermark)
        watermark = torch.where(watermark < (- eps), torch.zeros_like(watermark).cuda() - eps, watermark)

        # updata adv images
        a = images.data + watermark.data
        a = torch.clamp(a, min=0, max=1)
        watermark = a.data - images.data 
        
    return a
def low_level_target_PGD_sam_mid_to_mid(extractor,extractor2, images, guide_image, eps=0.1, alpha=1 / 255, iters=200,
                 l_norm =1,   rho=0.0001,skip=7,ratio=0.1):
    # basic settings
    
    # 0: l2-norm; 1: linf-norm

    # initialization
    #  for crafting adv images
    watermark = torch.zeros_like(images).cuda()

    #  for SAM
    s = images.clone()
    t = guide_image.clone()
    a = images.clone()
    rho_s = rho*ratio
    rho_t = rho*ratio
    rho_a = rho
    # rho_s2 = rho2*ratio2
    # rho_t2 = rho2*ratio2
    # rho_a2 = rho2
    conv1=extractor(norm_layer(augment_layer(s )))
    
    # the main loop
    for i in range(int(iters)):
        pert_s = torch.zeros_like(conv1).cuda()
        pert_t = torch.zeros_like(conv1).cuda()
        pert_a = torch.zeros_like(conv1).cuda()
        pert_s_input = torch.zeros_like(s).cuda()
        pert_t_input = torch.zeros_like(s).cuda()
        pert_a_input = torch.zeros_like(s).cuda()
        # the first step
        if i % skip == 0:  # perform sam
            # initialization
            pert_s.requires_grad_()
            pert_t.requires_grad_()
            pert_a.requires_grad_() 
            pert_s_input.requires_grad_()
            pert_t_input.requires_grad_()
            pert_a_input.requires_grad_() 
            
            # calculate loss
            sou_mid = extractor(norm_layer(augment_layer(s + pert_s_input)))
            tar_mid = extractor(norm_layer(augment_layer(t + pert_t_input)))
            adv_mid = extractor(norm_layer(augment_layer(a + pert_a_input)))
            sou_h_feats = extractor2( sou_mid  + pert_s  ) 
            tar_h_feats = extractor2( tar_mid  + pert_t ) 
            adv_h_feats = extractor2( adv_mid  + pert_a )
            loss= criterion(sou_h_feats, tar_h_feats, adv_h_feats)
            print("loss1:  "+ str(loss))
            # craft perturbation in SAM
            loss.backward()
            if l_norm == 0:
                grad_s = pert_s.grad 
                pert_s = rho_s * grad_s / (torch.sum(grad_s ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16) 
                grad_t = pert_t.grad
                pert_t = rho_t * grad_t / (torch.sum(grad_t ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16) 
                grad_a = pert_a.grad
                pert_a = rho_a * grad_a / (torch.sum(grad_a ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16)
                
                grad_s = pert_s_input.grad 
                pert_s_input = rho_s * grad_s / (torch.sum(grad_s ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16) 
                grad_t = pert_t_input.grad
                pert_t_input = rho_t * grad_t / (torch.sum(grad_t ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16) 
                grad_a = pert_a_input.grad
                pert_a_input = rho_a * grad_a / (torch.sum(grad_a ** 2, dim=(1,2,3), keepdim=True).sqrt()+1E-16)
            else:
                pert_s = torch.sign(pert_s.grad)*rho_s
                pert_t = torch.sign(pert_t.grad)*rho_t
                pert_a = torch.sign(pert_a.grad)*rho_a
                pert_s_input = torch.sign(pert_s_input.grad)*rho_s
                pert_t_input = torch.sign(pert_t_input.grad)*rho_t
                pert_a_input = torch.sign(pert_a_input.grad)*rho_a
        
        # the second step
        watermark.requires_grad_()
        if watermark.grad is not None:
            watermark.grad.data.fill_(0)
        extractor.zero_grad()

        # calculate loss
        sou_mid = extractor(norm_layer(augment_layer(s  - pert_s_input))).detach()
        tar_mid = extractor(norm_layer(augment_layer(t  - pert_t_input))).detach()
        adv_mid = extractor(norm_layer(augment_layer(s  - pert_a_input + watermark)))
        sou_h_feats = extractor2( sou_mid  - pert_s  ).detach()
        tar_h_feats = extractor2( tar_mid  - pert_t  ).detach()
        adv_h_feats = extractor2( adv_mid  - pert_a  )
        
        loss= criterion(sou_h_feats, tar_h_feats, adv_h_feats)
        loss.backward()
        print("loss2:  "+ str(loss))
        # update and clip
        grad = torch.sign(watermark.grad)
        watermark = watermark.detach() + alpha * grad
        watermark = torch.where(watermark > (+ eps), torch.zeros_like(watermark).cuda() + eps, watermark)
        watermark = torch.where(watermark < (- eps), torch.zeros_like(watermark).cuda() - eps, watermark)
        
        # updata adv images
        a = images.data + watermark.data
        a = torch.clamp(a, min=0, max=1)
        watermark = a.data - images.data 
        
    return a 

