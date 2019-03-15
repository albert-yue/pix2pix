"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

import os
import os.path

import numpy as np
import array
import h5py

HDF5_EXTENSIONS = [
    '.h5', '.hdf5',
]


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in HDF5_EXTENSIONS)


def load_data(filename, dataset_name='data'):
    """
        Reads an HDF5 file into its three channels, and provides the image size as well
        For grayscale, lower values are farther away

        Returns: (channels, size)
            channels (numpy array of 3 arrays of floats): the values of the channels of the image
            size (2-tuple of ints): dimensions of the image
    """
    if not is_hdf5_file(filename):
        raise ValueError("file must be of HDF5 format")

    with h5py.File(filename, 'r') as h5f:
        return h5f[dataset_name][:]


def make_dataset(dir, max_dataset_size=float("inf")):
    h5_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_hdf5_file(fname):
                path = os.path.join(root, fname)
                h5_files.append(path)
    return h5_files[:min(max_dataset_size, len(h5_files))]


def default_loader(path, dataset_name='data'):
    return load_data(path, dataset_name)


class SceneActionFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 HDF5s in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(HDF5_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
