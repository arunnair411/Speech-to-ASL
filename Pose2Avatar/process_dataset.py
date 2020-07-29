import torch
import argparse
import os
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torchvision
from tqdm import tqdm


class Pose2AvatarDataset(Dataset):
    def __init__(self, dataset_path='interpolation.dataset'):
        '''Only uses the first frame in the pair.
        Please load data another way when trying to use temporal frame information'''
        self.data = torch.load(dataset_path)
        self.pose = self.data['pose_pair']
        self.avatar = self.data['avatar_pair']

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):
        return self.pose[idx,:,:,:3].permute(2,0,1), self.avatar[idx,:,:,:3].permute(2,0,1)*(1.0/255)

def save_train_images_to_dir(data_loader, dir, subfolder):
    for i, (pose_img, avatar_img) in enumerate(tqdm(data_loader)):
        pos_path = os.path.join(dir, 'A', subfolder, '{}.png'.format(i))
        ava_path = os.path.join(dir, 'B', subfolder, '{}.png'.format(i))

        torchvision.utils.save_image(pose_img, pos_path)
        torchvision.utils.save_image(avatar_img, ava_path)

def save_inference_images_to_dir(data_loader, dir):
    for i, pose_img in enumerate(tqdm(data_loader)):
        pos_path = os.path.join(dir, '{}.png'.format(i))
        torchvision.utils.save_image(pose_img, pos_path)


if __name__ == "__main__":
    ''' Sample command:
        python3 process_dataset.py --mode=train --dataset_path=/data2/t-arnair/projects/Speech-to-ASL/Gloss2Avatar/small_temporalpair.dataset --out_dir=example_data
        python3 process_dataset.py --mode=train --dataset_path=/data2/t-arnair/projects/Speech-to-ASL/Gloss2Avatar/temporalpair.dataset --out_dir=pose2avatar_data
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True)
    parser.add_argument('--dataset_path', type=str, default='Gloss2Avatar/temporalpair.dataset')
    parser.add_argument('--out_dir', type=str, default='Pose2Avatar/pose2video_data/')
    parser.add_argument('--train_test_split', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)


    data = Pose2AvatarDataset(args.dataset_path)


    if args.mode == 'inference':
        # create the file structure that pix2pix likes
        data_loader = DataLoader(data, batch_size=None,
                num_workers=args.num_workers, pin_memory=True)
        save_inference_images_to_dir(data_loader, args.out_dir)

    else:
        # create the file structure that pix2pix likes
        for f in ['A', 'B']:
            fp = os.path.join(args.out_dir, f)
            if not os.path.isdir(fp):
                    os.mkdir(fp)            
            for g in ['train', 'val', 'test']:
                fp = os.path.join(args.out_dir, f, g)
                if not os.path.isdir(fp):
                    os.mkdir(fp)

        # Split data
        train_split = int(len(data) * args.train_test_split)
        val_split = int(len(data)*((1.0+args.train_test_split)/2))
        indices = list(range(len(data)))
        train_idx, val_idx, test_idx = indices[:train_split], indices[train_split:val_split], indices[val_split:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(data, batch_size=None, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(data, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(data, sampler=test_sampler,
            num_workers=args.num_workers, pin_memory=True)


        save_train_images_to_dir(train_loader, args.out_dir, 'train')
        save_train_images_to_dir(val_loader, args.out_dir, 'val')
        save_train_images_to_dir(test_loader, args.out_dir, 'test')


