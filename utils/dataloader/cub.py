import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Define the paths for the CUB dataset
THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

# Create a dataset for the CUB bird classification task
class CUB(Dataset):
    
    def __init__(self, setname, args):
        # Load the file paths and labels for the dataset split
        split_file = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(split_file, 'r').readlines()][1:]
        
        # Create lists to store the data and labels
        data = []
        label = []
        label_count = -1
        class_labels = []

        # Loop through each line of the split file
        for line in lines:
            # Extract the image file name and label from the line
            image_file, class_label = line.split(',')
            image_path = osp.join(IMAGE_PATH, image_file)
            
            # If the class label hasn't been seen before, add it to the list of labels
            if class_label not in class_labels:
                class_labels.append(class_label)
                label_count += 1
                
            # Add the image path and label to the data and label lists
            data.append(image_path)
            label.append(label_count)

        # Store the data and labels as attributes of the class
        self.data = data
        self.label = label
        self.num_classes = np.unique(np.array(label)).shape[0]

        # Define the image transforms for the dataset based on the model type
        if args.model_type == 'AmdimNet':
            # For AmdimNet, use a larger image size and normalization parameters
            resize_size = 146
            crop_size = 128
            interpolation_mode = 3
            normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        else:
            # For other models, use a smaller image size and different interpolation method
            resize_size = 84
            crop_size = 84
            interpolation_mode = PIL.Image.BICUBIC
            normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation_mode),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalization
        ])

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image and label for a given index
        image_path, label = self.data[idx], self.label[idx]
        image = self.transform(Image.open(image_path).convert('RGB'))
        return image, label
