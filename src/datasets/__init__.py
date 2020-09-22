import torch

from torch.utils.data import Dataset

# ~ never used
class MNISTDataset(Dataset):
    # ~ custom function
    # transform: any required processing to be applied on the sample, usually they are callable classes instead of simple functions so that parameters of the transform need not be passed everytime it’s called.
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.images = dataset.images
        self.labels = dataset.labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        data_sample = self.images[idx]
        label_sample = self.labels[idx]
        if self.transform:
            data_sample = self.transform(data_sample)
            label_sample = self.transform(label_sample)
        return data_sample, label_sample

    def next_batch(self, batch_size):
        image_batch, label_batch = self.dataset.next_batch(batch_size=batch_size)
        return torch.from_numpy(image_batch), torch.from_numpy(label_batch)
        # TODO(hwang): figure out why `ToTensor` caused error here
        #return self.transform(image_batch), self.transform(label_batch)

# ~ imported but never used
class Cifar10Dataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.images = dataset.images
        self.labels = dataset.labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        data_sample = self.images[idx]
        label_sample = self.labels[idx]
        if self.transform:
            data_sample = self.transform(data_sample)
            label_sample = self.transform(label_sample)
        return data_sample, label_sample

    def next_batch(self, batch_size):
        image_batch, label_batch = self.dataset.next_batch(batch_size=batch_size)
        return torch.from_numpy(image_batch), torch.from_numpy(label_batch)
        # TODO(hwang): figure out why `ToTensor` caused error here
        #return self.transform(image_batch), self.transform(label_batch)