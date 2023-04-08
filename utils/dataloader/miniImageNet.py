import os.path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(THIS_PATH, '..', '..'))
IMAGE_PATH = os.path.join(ROOT_PATH, 'data/miniimagenet/images')
SPLIT_PATH = os.path.join(ROOT_PATH, 'data/miniimagenet/split')

class MiniImageNet(Dataset):
    """Usage: 
    """
    def __init__(self, setname, args):
        csv_path = os.path.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(IMAGE_PATH, name)
            data.append(path)
            label.append(int(wnid))
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        # Transformation
        if args.model_type == 'ConvNet':
            image_size = 84
            transform_list = [transforms.Resize(92),
                              transforms.CenterCrop(image_size)]
            normalize_list = [np.array([0.485, 0.456, 0.406]),
                              np.array([0.229, 0.224, 0.225])]
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(*normalize_list))
            self.transform = transforms.Compose(transform_list)
        elif args.model_type == 'ResNet':
            image_size = 80
            transform_list = [transforms.Resize(92),
                              transforms.CenterCrop(image_size)]
            normalize_list = [np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])]
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(*normalize_list))
            self.transform = transforms.Compose(transform_list)
        elif args.model_type == 'AmdimNet':
            INTERP = 3
            post_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            transform_list = [transforms.Resize(146, interpolation=INTERP),
                              transforms.CenterCrop(128),
                              post_transform]
            self.transform = transforms.Compose(transform_list)
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
