# Error_TransFormer

## Create the environment
We provide the dockerfile in ~/Error_TransFormer/Dockerfile_adv. 
```

docker build -f Dockerfile_adv -t etf_images:0.1 .
nvidia-docker run -it --name ETF_container -v /home/:/home/ -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size 32G   xxxxxxxxxx ("xxxxxxxxxx" refer to IMAGE ID)

```
Please put the file Error_TransFormer in the home path. Or change the path below.

## Step one : Train lightweight surrogate model 

You can train the lightweight surrogate model by yourself. We also provide a *pre-trained model which is trained on the 1000 images* from ImageNet validation.(/home/Error_TransFormer/results/ETF-I/I-1.pth.tar) Utilizing the provided model, you can go straight to the step two.

Please provide the path of ILSVRC2012_img_val (--data_dir  )

```
#--img_sum                  The number of images
#--img_num                  The number of per categary
#--attack_method            options ETF_PGD  or  Deep_PGD 
#--lightweight_model        The location of model
#--attack                   ETF-I or ETF-C.  Train model using Instance classification or Contrastive learning.  Here, the pre-train model we previded is trained by ETF-I.
 
python /home/Error_TransFormer/Train_model.py --img_sum 1000   --img_num 1 --backbone resnet18 --batch_size 1000 \
                     --attack ETF-I \
                     --root_path  /home/Error_TransFormer   \
                     --lightweight_model  /home/Error_TransFormer/results/ETF-I/I-1.pth.tar  --seed 1  \
                     --data_dir             /home/ILSVRC2012_img_val
```
## Step two : Mount ETF Attacks 
We show the method of ETF_PGD, which attacks the shallow layers of the lightweight surrogate model. In contrast, you can use a PGD attack on the entire lightweight surrogate model using cross-entropy loss. 
```
# Attack the shallow layers of the surrogate model.
python /home/Error_TransFormer/ETF_Attack.py --backbone resnet18  --img_sum 1000   --img_num 1 --batch_size 100 \
                --root_path  /home/Error_TransFormer   \
                --lightweight_model  /home/Error_TransFormer/results/ETF-I/I-1.pth.tar   --seed 1  \
                --data_dir                /home/ILSVRC2012_img_val         \
                --attack_method ETF_PGD
                
# Attack the whole surrogate model.                
python /home/Error_TransFormer/ETF_Attack.py --backbone resnet18  --img_sum 1000   --img_num 1 --batch_size 100 \
                --root_path  /home/Error_TransFormer   \
                --lightweight_model  /home/Error_TransFormer/results/ETF-I/I-1.pth.tar   --seed 1  \
                --data_dir                /home/ILSVRC2012_img_val         \
                --attack_method Deep_PGD
```             
