# Import required libraries
import os
import os.path
import numpy as np
import random
import pickle
import math
import sys
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
_TIERED_IMAGENET_DATASET_DIR = 'data/tieredimagenet/'

# Define function to load data from pickle files
def load_data(file_path):
    try:
        # Try to open the file normally
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except:
        # If that doesn't work, open the file with latin1 encoding
        with open(file_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
    return data

# Define dictionary containing file paths for different dataset phases
file_paths = {
    'train': [os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_labels.pkl')],
    'val': [os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'val_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR,'val_labels.pkl')],
    'test': [os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_labels.pkl')]
}

# Define the tieredImageNet dataset class
class tieredImageNet(data.Dataset):
    def __init__(self, phase='train', data_aug=False):
        # Ensure that the provided phase is valid
        assert(phase=='train' or phase=='val' or phase=='test')
        
        # Get the file paths for the given phase
        image_path = file_paths[phase][0]
        label_path = file_paths[phase][1]

        # Load the label data from the given file path
        label_data = load_data(label_path)
        labels = label_data['labels']

        # Load the image data from the given file path
        self.data = np.load(image_path)['images']

        # Map the labels to integer values
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        # Set the label data for the dataset
        self.label = label

        # Set the number of classes in the dataset
        self.num_class = len(set(label))

        # Define normalization transform
        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        
        # Define the transformation pipeline for the dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        # Get the image and label data for the given index
        img, label = self.data[index], self.label[index]

        # Apply the transformation pipeline to the
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)