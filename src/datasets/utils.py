# ~ see https://discuss.pytorch.org/t/efficient-dataset-indexing/9221/4 for this sampler
# this file is used by "cyclic_worker.py"
import numpy as np

from torch.utils import data 
import torch
from torchvision import datasets, transforms

# ~ torch.utils.data.Sampler classes are used to specify the sequence of indices/keys used in data loading. They represent iterable objects over the indices to datasets. 
# E.g., in the common case with stochastic gradient decent (SGD), a Sampler could randomly permute a list of indices and yield each one at a time, 
# or yield a small number of them for mini-batch SGD. A sequential or shuffled sampler will be automatically constructed based on the shuffle argument to a DataLoader. 
# Alternatively, users may use the sampler argument to specify a custom Sampler object that at each time yields the next index/key to fetch.
class DynamicSampler(object):
    def __init__(self, max_size=100):
        self.next_batch = [0]
        self.max_size = max_size

    def select_sample(self, indList):
        self.next_batch = indList

    def __iter__(self):
        return iter(self.next_batch)

    def __len__(self):
        return self.max_size

# ~ from https://discuss.pytorch.org/t/efficient-dataset-indexing/9221/4
def get_batch(dataset, indices=None):
    sampler = DynamicSampler(len(indices))
    loader = data.DataLoader(dataset, 
                  batch_size=len(indices), 
                  sampler=sampler) # this DataLoader uses the custom sampler defined above
    
    sampler.select_sample(indices)

    return iter(loader).next()

if __name__ == '__main__':
    # ~ see https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457 for the normalization numbers
    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))
    indices = np.arange(_i, _i+_BATCH_SIZE)
    batch = get_batch(train_dataset, indices)
    print(batch[0].size())
    print(batch[1].size())