import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import torchvision
import os
import pdb
class AugmentedDataset(Dataset):
    def __init__(self, dataset,image_augmented_num=1):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        self.image_augmented_num=image_augmented_num
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)
   
        for i in range(self.image_augmented_num):

            sample['image_augmented'+str(i)] = self.augmentation_transform['t'+str(i+1)](image)  #这样写，哪怕扩充图像是一个，也不会有问题
            
        return sample
 
if __name__ == '__main__':
    # One way to prepare 'data/selected_data.csv'
    """JPEG"""
    csv_path='selected_data_my.csv'
    os.system("touch "+csv_path)
    selected_data_csv = open(csv_path, 'w')
    data_writer = csv.writer(selected_data_csv)
    """PNG"""
    csv_path='./csv/test_data.csv'
    os.system("touch "+csv_path)
    test_data_csv = open(csv_path, 'w')
    test_data_writer = csv.writer(test_data_csv)
    
    dataset_dir = '/home/common/sunch/ILSVRC2012_img_val'
    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    label_ind = torch.randperm(1000).numpy()
    selected_labels_ls = np.array(dataset.classes)[label_ind]
    for label_name in selected_labels_ls: 
        data_writer.writerow([label_name]+os.listdir(os.path.join(dataset_dir, label_name)))
        test_data_writer.writerow([label_name]+[i.replace('.JPEG','.PNG') for i in os.listdir(os.path.join(dataset_dir, label_name))])
 