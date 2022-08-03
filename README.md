# Error_TransFormer

## Create the environment
```

docker build -f Dockerfile_adv -t etf_images:0.1 .
nvidia-docker run -it --name ETF_container -v /home/:/home/ -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size 32G   xxxxxxxxxx ("xxxxxxxxxx" refer to IMAGE ID)

```


## Step one : Train lightweight surrogate model 
```
img_sum                 # The number of images
img_num                 # The number of per categary
attack_method           # options ETF_PGD  or  Deep_PGD 
lightweight_model       # The location of model
attack                  # ETF-I or ETF-C.  Train model using Instance classification or Contrastive learning
```
You can train the lightweight surrogate model by yourself. We also provide a pre-trained model which is trained on the 1000 images from ImageNet validation.

```
python /home/Error_TransFormer/Train_model.py --img_sum 1000   --img_num 1 --backbone resnet18 --batch_size 1000 \
                     --attack ETF-I \
                     --root_path  /home/Error_TransFormer   \
                     --lightweight_model  /home/Error_TransFormer/results/ETF-I/I_test.pth.tar  --seed 1  \
                     --data_dir             /home/ILSVRC2012_img_val
```
## Step two : Mount ETF Attacks 
```
python /home/Error_TransFormer/ETF_Attack.py --backbone resnet10  --img_sum 1000   --img_num 1 --batch_size 100 \
                --root_path  /home/Error_TransFormer   \
                --lightweight_model  /home/Error_TransFormer/results/ETF-I/I-1.pth.tar   --seed 1  \
                --data_dir                /home/ILSVRC2012_img_val         \
                --attack_method ETF_PGD
```             
