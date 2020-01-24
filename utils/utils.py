import numpy as np
import random
from skimage.transform import resize
from scipy.io import loadmat
import pdb

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Microscope dataset."""
    def __init__(self, mat_paths, transform=None):
        """
        Args:
            mat_paths (list of strings): List of paths to the data mat files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Outputs:
            sample (dict): Sample of our dataset will be a dict {'input_data': input_data, 'target_output': target_output,
            'file_name': self.paths[index], 'file_name_idx':file_name_idx}
        """        
        self.paths = mat_paths
        self.transform = transform
    
    def __getitem__(self, index):
        try:
            mat_contents = loadmat(self.paths[index]) # Set to False if it throws an error
        except ValueError:
            mat_contents = loadmat(self.paths[index], verify_compressed_data_integrity=False)
            print('File {} had a decompression error'.format(self.paths[index]))
        
        input_data = mat_contents['input_data'].astype('float32')
        target_output = mat_contents['target_output'].astype('float32')
        file_name_idx = mat_contents['file_name_idx'].astype('float32')
        sample = {'input_data': input_data, 'target_output': target_output, 
          'file_name': self.paths[index], 'file_name_idx':file_name_idx}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.paths)

class RandomHorizontalFlipArun(object):
    """Flip the data in a sample

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample (dict): A dictionary containing our data. {'channel_data': 
            curr_channel_data, 'true_segmentation': curr_GT, 'reference_B': curr_B}

        Returns:
            sample (dict): Dictionary with randomly flipped components.
        """
        if random.random() < self.p:            
            # adding the copy to solve the pytorch error
            sample = {'input_data': np.flip(sample['input_data'], -1).copy(),
                'target_output': np.flip(sample['target_output'], -1).copy(),
                'file_name': sample['file_name'], 'file_name_idx': sample['file_name_idx']} 
        return sample
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_data, target_output, file_name, file_name_idx = sample['input_data'],\
            sample['target_output'], sample['file_name'], sample['file_name_idx']
        # Note: Don't need to do the below... but so elegant!
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'input_data': torch.from_numpy(input_data), 
                'target_output': torch.from_numpy(target_output),
                'file_name': file_name,
                'file_name_idx': file_name_idx}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# PyTorch Dataset class written for the FAIR fastMRI data
# class SliceData(Dataset):
#     """
#     A PyTorch Dataset that provides access to MR image slices.
#     """

#     def __init__(self, root, transform, challenge, sample_rate=1):
#         """
#         Args:
#             root (pathlib.Path): Path to the dataset.
#             transform (callable): A callable object that pre-processes the raw data into
#                 appropriate form. The transform function should take 'kspace', 'target',
#                 'attributes', 'filename', and 'slice' as inputs. 'target' may be null
#                 for test data.
#             challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
#             sample_rate (float, optional): A float between 0 and 1. This controls what fraction
#                 of the volumes should be loaded.
#         """
#         if challenge not in ('singlecoil', 'multicoil'):
#             raise ValueError('challenge should be either "singlecoil" or "multicoil"')

#         self.transform = transform
#         self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
#             else 'reconstruction_rss'

#         self.examples = []
#         files = list(pathlib.Path(root).iterdir())
#         if sample_rate < 1:
#             random.shuffle(files)
#             num_files = round(len(files) * sample_rate)
#             files = files[:num_files]
#         for fname in sorted(files):
#             kspace = h5py.File(fname, 'r')['kspace']
#             num_slices = kspace.shape[0]
#             self.examples += [(fname, slice) for slice in range(num_slices)]

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         fname, slice = self.examples[i]
#         with h5py.File(fname, 'r') as data:
#             kspace = data['kspace'][slice]
#             target = data[self.recons_key][slice] if self.recons_key in data else None
#             return self.transform(kspace, target, data.attrs, fname.name, slice)

# -------------------------------------------------------------------------------------------------
# Data transformer used for the FAIR training dataset - a similar one written for the FAIR test set
# class DataTransform:
#     """
#     Data Transformer for training U-Net models.
#     """

#     def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
#         """
#         Args:
#             mask_func (common.subsample.MaskFunc): A function that can create a mask of
#                 appropriate shape.
#             resolution (int): Resolution of the image.
#             which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
#             use_seed (bool): If true, this class computes a pseudo random number generator seed
#                 from the filename. This ensures that the same mask is used for all the slices of
#                 a given volume every time.
#         """
#         if which_challenge not in ('singlecoil', 'multicoil'):
#             raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
#         self.mask_func = mask_func
#         self.resolution = resolution
#         self.which_challenge = which_challenge
#         self.use_seed = use_seed

#     def __call__(self, kspace, target, attrs, fname, slice):
#         """
#         Args:
#             kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
#                 data or (rows, cols, 2) for single coil data.
#             target (numpy.array): Target image
#             attrs (dict): Acquisition related information stored in the HDF5 object.
#             fname (str): File name
#             slice (int): Serial number of the slice.
#         Returns:
#             (tuple): tuple containing:
#                 image (torch.Tensor): Zero-filled input image.
#                 target (torch.Tensor): Target image converted to a torch Tensor.
#                 mean (float): Mean value used for normalization.
#                 std (float): Standard deviation value used for normalization.
#                 norm (float): L2 norm of the entire volume.
#         """
#         target = transforms.to_tensor(target)
#         kspace = transforms.to_tensor(kspace)
#         # Apply mask
#         seed = None if not self.use_seed else tuple(map(ord, fname))
#         masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
#         # Inverse Fourier Transform to get zero filled solution
#         image = transforms.ifft2(masked_kspace)
#         # Crop input image to given resolution if larger
#         smallest_width = min(min(args.resolution, image.shape[-2]), target.shape[-1])
#         smallest_height = min(min(args.resolution, image.shape[-3]), target.shape[-2])
#         crop_size = (smallest_height, smallest_width)
#         image = transforms.complex_center_crop(image, crop_size)
#         target = transforms.center_crop(target, crop_size)

#         # Absolute value
#         image = transforms.complex_abs(image)
#         # Apply Root-Sum-of-Squares if multicoil data
#         if self.which_challenge == 'multicoil':
#             image = transforms.root_sum_of_squares(image)
#         # Normalize input
#         image, mean, std = transforms.normalize_instance(image, eps=1e-11)
#         image = image.clamp(-6, 6)

#         # Normalize target
#         target = transforms.normalize(target, mean, std, eps=1e-11)
#         target = target.clamp(-6, 6)
#         return image, target, mean, std, attrs['norm'].astype(np.float32)