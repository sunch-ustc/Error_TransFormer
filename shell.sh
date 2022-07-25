  
"""
img_sum                 # The number of images
img_num                 # The number of per categary

attack_method           # options ETF_PGD  or  Deep_PGD 
pretext_checkpoint      # The location of model
"""

"""Train_model"""
# python /home/common/sunch/Error_TransFormer/Train_model.py --img_sum 1000   --img_num 1 --backbone resnet10 --batch_size 1000 \
#                     --attack ETF-I \
#                     --pretext_dir         /home/common/sunch/Error_TransFormer/results/ETF-C \
#                     --pretext_checkpoint  /home/common/sunch/Error_TransFormer/results/ETF-I/I_test.pth.tar  --seed 1 
#                     --data_dir             /home/common/sunch/ILSVRC2012_img_val
 
"""ETF_Attack"""
python /home/common/sunch/Error_TransFormer/ETF_Attack.py --backbone resnet10  --img_sum 500   --img_num 1 --batch_size 100 \
                --path_save_adv_image     /home/common/sunch/Error_TransFormer/images/image_adv \
                --path_feature_extractor  /home/common/sunch/Error_TransFormer/results/ETF-I/I-1.pth.tar   --seed 1  \
                --data_dir                /home/common/sunch/ILSVRC2012_img_val         \
                --attack_method ETF_PGD
 
 