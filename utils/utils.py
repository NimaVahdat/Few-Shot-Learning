import torch
import numpy as np

def set_device(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    return device

def ensure_directory(directory_path, remove_existing=False):
    if os.path.exists(directory_path):
        if remove_existing:
            shutil.rmtree(directory_path)
            os.mkdir(directory_path)
    else:
        os.makedirs(directory_path)

def euclidean_distance(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    distance = ((a - b)**2).sum(dim=2)
    return -distance
