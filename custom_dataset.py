# Reference: https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data
# Auther : Hyunki Seong, hynkis@kaist.ac.kr

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

import cv2

class customDataset(Dataset):
    """
    Load Image and label dataset
    """
    def __init__(self, dataset_dir, img_name_txt, label_txt, transform=None):
        # Path of root dataset dir
        self.dataset_root_dir = dataset_dir

        # list of images to load in a .txt file
        #   note that in the .txt file the image names are stored with the extension(.jpg or .png)
        self.images = open(img_name_txt, "rt").read().split("\n")[:-1]
        # Apply transform if you need
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image data at every iteration
        img_name = self.images[index]
        image_path = os.path.join(self.dataset_root_dir, img_name)
        image = self.load_image(path=image_path)

        # Load label data at eveery iteration
        label = np.array([-1])

        data = {
                'image': torch.FloatTensor(image), # (3,224,224)
                'label': torch.FloatTensor(label),
                }

        return data

    def load_image(self, path=None):
        # can use any other library too like OpenCV as long as you are consistent with it
        raw_image = Image.open(path) # RGB image
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0)) # (H,W,C) to (C,W,H)
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def custom_pil_imshow(self, pil_img):
        cv_img = np.array(pil_img)
        # Convert RGB to BGR
        cv_img = cv_img[:,:,::-1].copy()
        cv2.imshow("img", cv_img)
        cv2.waitKey()

    def custom_torch_imshow(self, torch_img):
        np_img = torch_img.numpy().transpose(2, 1, 0) # ndarray is 0~1 for cv imshow, not 0~255
        cv_img = np_img[:,:,::-1].copy() # Convert RGB to BGR
        cv2.imshow("img", cv_img)
        cv2.waitKey()

def main():
    dataset_dir  = "./dataset/valid/img_data"
    img_name_txt = "./dataset/valid/img_data.txt"
    label_txt    = "./dataset/valid/img_data.txt"
    transform    = None

    BATCH_SIZE = 10
    SHUFFLE_BATCH = True

    dataset = customDataset(dataset_dir=dataset_dir,
                            img_name_txt=img_name_txt,
                            label_txt=label_txt,
                            transform=transform,
                            )
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_BATCH,
                            num_workers=4,
                            drop_last=True
                            )

    for i, data in enumerate(dataloader):
        print("===== %d th batch in dataloader =====" %(i))
        print(data["image"].size())  # input image
        print(data["label"].size())  # class label
        for j in range(BATCH_SIZE):
            print("---------- %d th image in a batch" %(j))
            dataset.custom_torch_imshow(data["image"][j])

        if i > 10:
            break


if __name__ == '__main__':
    main()