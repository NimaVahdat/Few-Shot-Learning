import torch
import numpy as np

def set_device(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    return device

def ensure_directory(directory_path, remove_existing=False):
    """
    Creates the directory at the given path if it does not exist, otherwise removes it
    and creates it again if the remove_existing flag is set to True.
    :param path: the path of the directory to be created
    :param remove_existing: whether to remove the directory if it already exists
    """
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

class Averager():
    """
    Computes the running average of a sequence of values
    """

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
def compute_accuracy(logits, label):
    """
    Computes the accuracy of a prediction given the logits and the label
    :param logits: the predicted logits
    :param label: the ground truth label
    :return: the accuracy of the prediction
    """
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():
    """
    Measures the time taken by an operation
    """

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()
def pretty_print(x):
    """
    Pretty prints the input x
    :param x: the object to be pretty printed
    """
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Computes the 95% confidence interval for an array of mean accuracy (or mAP) values
    :param data: the array of mean accuracy (or mAP) values
    :return: the 95% confidence interval for this data
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m
