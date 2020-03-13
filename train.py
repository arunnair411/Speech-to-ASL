# -------------------------------------------------------------------------------------------------
# Created on:                                   2020-01-02
# Last edited on:                               2020-01-02
# This is the training code 
# This version of the code just uses the U-Net encoder network I used for US
# -------------------------------------------------------------------------------------------------

# Low priority TODO:
# 0) Port visualization to testing code...


# # Sampling=0.20
# CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu-ids 0 1 --dataset=20200102_Sim1 --save-test-val-results --criterion=dscloss --test-rows=128 --test-cols=128 --epochs=40 --adam-lr=3e-4 --log-interval=100 --store-dir=20200102_20200102_Sim1 --test-batch-size=1024 --n-channels=2
# For testing - CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu-ids 0 1 --dataset=20200102_Sim1 --test-mode --save-test-val-results --load-g-model=checkpoints/20200102_20200102_Sim1/CP040_G_valDice0.970_PSNR38.219.pth --test-rows=128 --test-cols=128  --store-dir=20200103_20200102_Sim1_testOnly --test-batch-size=1024 --n-channels=2

# # Sampling=0.10
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpu-ids 0 1 2 3 --dataset=20200102_Sim2_0.10 --save-test-val-results --criterion=dscloss --test-rows=128 --test-cols=128 --epochs=40 --adam-lr=3e-4 --log-interval=100 --store-dir=20200107_20200102_Sim2 --test-batch-size=1024 --n-channels=2
# For testing - CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu-ids 0 1 --dataset=20200102_Sim2_0.10 --test-mode --save-test-val-results --load-g-model=checkpoints/20200107_20200102_Sim2/CP040_G_valDice0.961_PSNR36.534.pth --test-rows=128 --test-cols=128  --store-dir=20200107_20200102_Sim2_testOnly --test-batch-size=1024 --n-channels=2

# # Sampling=0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpu-ids 0 1 2 3 --dataset=20200102_Sim3_0.05 --save-test-val-results --criterion=dscloss --test-rows=128 --test-cols=128 --epochs=40 --adam-lr=3e-4 --log-interval=100 --store-dir=20200107_20200102_Sim3 --test-batch-size=1024 --n-channels=2
# For testing - CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu-ids 0 1 --dataset=20200102_Sim3_0.05 --test-mode --save-test-val-results --load-g-model=checkpoints/20200107_20200102_Sim2/CP040_G_valDice0.961_PSNR36.534.pth --test-rows=128 --test-cols=128  --store-dir=20200107_20200102_Sim2_testOnly --test-batch-size=1024 --n-channels=2


# After code revamp - 2020-01-14
# # Sampling=0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --gpu-ids 0 1 2 3 --dataset=20200102_Sim3_0.05  --store-dir=20200114_20200102_Sim3_l1 --save-test-val-results --n-channels=2 --criterion-g=l1loss --epochs=200 --batch-size=64 --test-batch-size=1024 --adam-lr=3e-4 --lr-step-size=40 --lr-gamma=0.5 --test-rows=128 --test-cols=128 --log-interval=10
# For testing - CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu-ids 0 1 --dataset=20200102_Sim3_0.05 --test-mode --save-test-val-results --load-g-model=checkpoints/20200107_20200102_Sim2/CP040_G_valDice0.961_PSNR36.534.pth --test-rows=128 --test-cols=128  --store-dir=20200107_20200102_Sim2_testOnly --test-batch-size=1024 --n-channels=2

# After code revamp - 2020-01-16
# # Sampling=0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --gpu-ids 0 1 2 3 --dataset=20200102_Sim3_0.05  --store-dir=20200116_20200102_Sim3_l1 --fair --in-chans=2 --normalization=batchnorm --criterion-g=l1loss --epochs=200 --batch-size=64 --test-batch-size=1024 --adam-lr=3e-4 --lr-step-size=40 --lr-gamma=0.5 --test-rows=128 --test-cols=128

# After code revamp - 2020-01-20
# # Sampling=0.05
# CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --dataset=20200102_Sim3_0.05  --store-dir=20200120_20200102_Sim3_l1 --in-chans=2 --normalization=batchnorm --criterion-g=l1loss --epochs=10 --batch-size=64 --test-batch-size=256 --adam-lr=3e-4 --lr-step-size=40 --lr-gamma=0.5 --test-rows=128 --test-cols=128 --no-parallel

import numpy as np
import random
import logging
import argparse, pathlib
import os, sys, glob
import time
import pdb
import shutil

# PyTorch DNN imports
import torch
import torch.nn
import torch.optim
import torchvision 

# U-Net related imports
from unet import GeneratorUnet1_1, GeneratorUnet1_1_FAIR, visualize_neurons

# Loss Function Imports
from utils import DiceCoeffLoss

# Tensorboard import
from torch.utils.tensorboard import SummaryWriter

# Dataset imports
from utils import retrieve_dataset_filenames

# Dataloader imports
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import MyDataset, RandomHorizontalFlipArun, ToTensor

# Validation/Testing data related inputs
from utils import eval_net

# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# # Section I - train, evaluate, and visualize functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Training function for one epoch
def train_epoch(params, epoch, g_net, criterion_g, train_loader, optimizer_G, scheduler, writer):
    print('-'*80)
    print('Set the neural network to training mode...')
    print('-'*80)
    
    # Need to do this to ensure batchnorm and dropout layers exhibit training behavior
    g_net.train() 

    # Initialize the loss for this epoch to 0
    g_epoch_loss = 0.
    # avg_loss = 0. - FAIR code


    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader) # len(train_loader) is the number of batches to process in one training epoch

    tot_epochs = params['epochs']

    for iterIdx, sample in enumerate(train_loader):
        input_data = sample['input_data']
        target_output = sample['target_output']

        # Copy the data to GPU
        input_data = input_data.to(params['device'])
        target_output = target_output.to(params['device'])
        target_output_flat = target_output.view(-1) # Flatten (i.e.) vectorize the data

        # Pass the minibatch through the network to get predictions
        preds = g_net(input_data)
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(-1)            

        # Calculate the loss for the batch
        loss = criterion_g(preds_flat, target_output_flat)

        # Zero the parameter gradients
        optimizer_G.zero_grad()

        # Compute the gradients and take a step    
        loss.backward()            
        optimizer_G.step()
        
        g_epoch_loss += loss.item() # Note: calling .item() creates a a copy which is stored in g_epoch_loss... so modifying the latter doesn't modify the former. Verified!
        # avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item() - FAIR - dunno what is going on here...
        writer.add_scalar('TrainLoss', loss.item(), global_step + iterIdx)

        if iterIdx % params['log_interval'] == 0:
            # logging.info(
            #     f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            #     f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            #     f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
            #     f'Time = {time.perf_counter() - start_iter:.4f}s',
            # )
            logging.info(
                f'Epoch = [{epoch+1:3d}/{ tot_epochs:3d}] '
                f'Iter = [{iterIdx:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {g_epoch_loss/(iterIdx+1):.4g} '
                f'MiniBatch Time = {time.perf_counter() - start_iter:.4f}s',
            )                    
        start_iter = time.perf_counter()
    # scheduler.step(epoch) # Wasn't sure if it should be epoch or epoch+1, so just doing step() - step has well defined behavior
    scheduler.step()

    # ReDefine avg_loss to what makes more sense to me instead of the FAIR code
    avg_loss = g_epoch_loss/(iterIdx+1)

    return avg_loss, time.perf_counter() - start_epoch

# -------------------------------------------------------------------------------------------------
## Evaluation function for val/test data
def evaluate(params, dataset_deets, epoch, g_net, data_loader, writer, data_string, req_filenames):

    print('-'*80)
    print('Set the neural network to testing mode...')
    print('-'*80)
    # Set the network to eval mode to freeze BatchNorm weights
    g_net.eval()
    start = time.perf_counter()
    _dice,  _PSNR  = eval_net(g_net, req_filenames, params, dataset_deets, data_string, data_loader)
    print(f"{data_string} DSC Score  (Higher is better): {_dice}")
    print(f"{data_string} PSNR Score (Higher is better): {_PSNR}")

    writer.add_scalar(f"Loss/{data_string}_dsc",  _dice, epoch)
    writer.add_scalar(f"Loss/{data_string}_psnr", _PSNR, epoch)

    mean_points = _PSNR
    # return np.mean(losses), time.perf_counter() - start
    return mean_points, time.perf_counter() - start

# -------------------------------------------------------------------------------------------------
## Visualization function for network output -- track it using tensorboard
def visualize(params, epoch, g_net, data_loader, writer):
    def save_image(image, tag, nrow):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=nrow, pad_value=1)
        writer.add_image(tag, grid, epoch)

    g_net.eval()
    with torch.no_grad():
        for iterIdx, sample in enumerate(data_loader):

            # input, target, mean, std, norm = data
            # input = input.unsqueeze(1).to(args.device)
            # target = target.unsqueeze(1).to(args.device)
            # output = model(input)

            input_data = sample['input_data'].to(params['device']) 
            target_output = sample['target_output'].to(params['device']) 
            
            preds = g_net(input_data)
            preds = torch.sigmoid(preds)

            save_image(target_output, 'Images/Target', nrow=8)
            save_image(preds, 'Images/Reconstruction', nrow=8)
            save_image(torch.abs(target_output - preds), 'Images/Error', nrow=8)
            break

# -------------------------------------------------------------------------------------------------
# # Section II - Dataset definition and dataloader creation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Create dataset objects to load the data
def create_datasets(params):
    print('-'*80)
    print('Retrieve dataset details...')
    print('-'*80)
    dataset_deets = retrieve_dataset_filenames(dataset=params['dataset'])
    
    print('-'*30)
    print('Reading in data filenames...')
    print('-'*30)
    split_train = []
    for data_path in dataset_deets['train_data_path_list']:
        curr_train_data_filenames = sorted(glob.glob(os.path.join(data_path,'*.mat')))
        split_train.extend(curr_train_data_filenames)
    split_validation = []
    for data_path in dataset_deets['val_data_path_list']:
        curr_val_data_filenames = sorted(glob.glob(os.path.join(data_path,'*.mat')))
        split_validation.extend(curr_val_data_filenames)    
    split_test = []
    for data_path in dataset_deets['test_data_path_list']:
        curr_test_data_filenames = sorted(glob.glob(os.path.join(data_path,'*.mat')))
        split_test.extend(curr_test_data_filenames)
    
    # If there is no training data
    if not split_train:
        print('Trying to train on a test set. Exiting.')
        sys.exit(0)
    if params['trunc_data_flag']:
        # Truncate the dataset to be more manageable - use this to check if neural network code is right
        # It should overfit to the reduced size data and give near 100% results...
        split_train = split_train[0:1024]
        split_validation = split_validation[0:1024]
        split_test = split_test[0:1024]                

    # train_data = SliceData(
    #     root=args.data_path / f'{args.challenge}_train',
    #     transform=DataTransform(train_mask, args.resolution, args.challenge),
    #     sample_rate=args.sample_rate,
    #     challenge=args.challenge
    # )
    # dev_data = SliceData(
    #     root=args.data_path / f'{args.challenge}_val',
    #     transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
    #     sample_rate=args.sample_rate,
    #     challenge=args.challenge,
    # )

    train_data = MyDataset(mat_paths=split_train,
                            transform=transforms.Compose([
                                RandomHorizontalFlipArun(params['to_flip']/2), #params['to_flip']/2 is 0 or 0.5 - exactly what we want!
                                ToTensor()
                            ]))
    val_data = MyDataset(mat_paths=split_validation,
                            transform=transforms.Compose([
                                RandomHorizontalFlipArun(0), #params['to_flip']/2 is 0 or 0.5 - exactly what we want!
                                ToTensor()
                            ]))
    test_data = MyDataset(mat_paths=split_test,
                            transform=transforms.Compose([
                                RandomHorizontalFlipArun(0), #0 - we want a 0 probability of flipping for the test set
                                ToTensor()
                            ]))
    req_filenames_dict = {'train': split_train, 'val': split_validation, 'test':split_test}
    return train_data, val_data, test_data, dataset_deets, req_filenames_dict

# -------------------------------------------------------------------------------------------------
## Create dataloaders to load data from my dataset objects
def create_data_loaders(params):
    train_data, val_data, test_data, dataset_deets, req_filenames_dict = create_datasets(params)
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)] 
    display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 64)] 

    train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'],
                                shuffle=True, num_workers= 2*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    val_loader   = DataLoader(dataset=val_data, batch_size=params['test_batch_size'],
                                num_workers= 2*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files                                                                           
    test_loader  = DataLoader(dataset=test_data, batch_size=params['test_batch_size'],
                                num_workers= 2*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    # train_loader = DataLoader(
    #     dataset=train_data,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    # dev_loader = DataLoader(
    #     dataset=dev_data,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    # display_loader = DataLoader(dataset=display_data, batch_size=16, num_workers=2*len(params['gpu_ids']), pin_memory=True)

    display_loader = DataLoader(dataset=display_data, batch_size=64, num_workers=2*len(params['gpu_ids']), pin_memory=True)
    
    return train_loader, val_loader, test_loader, display_loader, dataset_deets, req_filenames_dict


# -------------------------------------------------------------------------------------------------
# # Section III - Argument parser and argument dictionary creation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Argument parser
def parse_args(args):
    parser = argparse.ArgumentParser(description='PyTorch Microscope Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Parameters to update every run (or thereabouts)
    # IMPORTANT: usage --gpu-ids 0 1 2 3
    parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, 
                        help='List of GPU-IDs to run code on')
    # IMPORTANT: usage --dataset=anechoic+hyp
    parser.add_argument('--dataset', required=True, type=str,
                        help='<predefined dataset name>|<path to dataset>')                        
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10, metavar='CI',
                        help='Once in how many epochs to save the model file')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in mat files')
    ## Parameters to update less often
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')    
    parser.add_argument('--checkpoint', type=str, default=False,
                        help='Path to an existing checkpoint. Used along with "--resume"')    
    parser.add_argument('--prng-seed', type=int, default=1337, metavar='S',
                        help='Seed for all the pseudo-random number generators')
    parser.add_argument('--fair', action='store_true', default=False,
                        help='Flag to use the fastMRI UNet from FAIR')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help='Flag to prevent paralellization of the model across the GPUs')
    parser.add_argument('--in-chans', default=2, type=int, metavar='IC',
                        help='Number of input channels')
    parser.add_argument('--chans', default=64, type=int, metavar='CH',
                        help='Number of output channels of the first convolution layer')                        
    parser.add_argument('--normalization', choices=['none','batchnorm','instancenorm'], default='batchnorm', type=str,
                        help='none|batchnorm|instancenorm')                                                
    parser.add_argument('--criterion-g', choices=['dscloss','l2loss','l1loss','bceloss'], required=True,
                        help='dscloss|l2loss|l1loss|bceloss')                        
    # IMPORTANT: usage --epochs=80 or --epochs 80 (both work)
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='Number of epochs to train (default: 80)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='Input mini-batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1000)')                        
    parser.add_argument('--trunc-data-flag', action='store_true', default=False,
                        help='Work with truncated dataset')
    parser.add_argument('--adam-lr', type=float, default=0.00001, metavar='LR',
                        help='Adam learning rate (default: 1e-5)')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--flip-training', action='store_true', default=False,
                        help='Whether to flip the training data')
    parser.add_argument('--test-rows', type=int, default=128, metavar='R',
                        help='Number of rows in the test data inputs (default: 128)')
    parser.add_argument('--test-cols', type=int, default=128, metavar='C',
                        help='Number of cols in the test data inputs (default: 128)')                                 
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many mini-batches to wait before logging training status')    
                                                
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']         = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']         = parsed_args.dataset                     # Which dataset to processs
    params['checkpoints_dir'] = os.path.join("checkpoints",parsed_args.store_dir)   # Directory in which to store the model checkpoints and training details as .pt files
    params['results_dir']     = os.path.join("results",parsed_args.store_dir)       # Directory in which to store the output images as mat files
    params['checkpoint_interval'] = parsed_args.checkpoint_interval       # Saves a checkpoint once every params['checkpoint_iterval'] epochs
    params['save_test_val_results'] = parsed_args.save_test_val_results # Whether to save val and test outputs as .mat files
    params['resume']          = parsed_args.resume                      # Flag set to true if training is to be resumed from provided checkpoint
    params['checkpoint']      = parsed_args.checkpoint                  # Directory from which the model is being loaded if its being loaded; false otherwise
    params['fair']            = parsed_args.fair                        # Flag to use the fastMRI UNet from FAIR
    params['no_parallel']     = parsed_args.no_parallel                 # Flag to prevent model paralellization across GPUs
    params['in_chans']        = parsed_args.in_chans                    # Number of input channels - defaults to 2 
    params['chans']           = parsed_args.chans                       # Number of output channels of the first convolution layer - defaults to 64
    params['normalization']   = parsed_args.normalization               # Set to 'none', 'batchnorm', or 'instancenorm'
    params['criterion_g']     = parsed_args.criterion_g                 # DNN loss function
    params['epochs']          = parsed_args.epochs                      # Total number of training epochs i.e. complete passes of the training data
    params['batch_size']      = parsed_args.batch_size                  # Number of training files in one mini-batch
    params['test_batch_size'] = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    params['trunc_data_flag'] = parsed_args.trunc_data_flag             # Flag on whether to truncate the dataset
    params['adam_lr']         = parsed_args.adam_lr                     # Learning rate for the Adam Optimzer     
    params['lr_step_size']    = parsed_args.lr_step_size                # Period of learning rate decay - after this number of epochs, the lr will decrease by the multiplicative factor of lr_gamma
    params['lr_gamma']        = parsed_args.lr_gamma                    # Multiplicative factor of learning rate decay. Default: 0.1.    
    params['to_flip']         = int(parsed_args.to_flip)                # Whether to augment the training data through flipping it laterally
    params['test_rows']       = parsed_args.test_rows                   # Number of rows in the test images
    params['test_cols']       = parsed_args.test_cols                   # Number of columns in the test images
    params['log_interval']    = parsed_args.log_interval                # How often to print loss information for a minibatch
    params['device']          = device                                  # Device to run the code on

    # Create a directory to store checkpoints if it doesn't already exist
    os.makedirs(params['checkpoints_dir'], exist_ok=True)

    return params


# -------------------------------------------------------------------------------------------------
# # Section IV - Smaller Utility Functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Seed all the random number generators
def seed_prng(parsed_args, device):
    # Do the below to ensure reproduceability - from the last comment at
    # https://discuss.pytorch.org/t/random-seed-initialization/7854/18
    # NOTE: I didn't do the num_workers thing they suggested, but reproducibility
    # was obtained without it
    # NOTE: According to FAIR fastMRI code, the three below lines will suffice for reproducibility... it doesn't set the CUDA seeds.
    np.random.seed(parsed_args.prng_seed)
    random.seed(parsed_args.prng_seed)
    torch.manual_seed(parsed_args.prng_seed)

    # if you are using GPU # This might not be necessary, as per https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
    if 'cuda' in device.type:
        print(f"Using the following GPUs: {parsed_args.gpu_ids}")
        torch.cuda.manual_seed(parsed_args.prng_seed)
        torch.cuda.manual_seed_all(parsed_args.prng_seed) 
        torch.backends.cudnn.enabled = True # This was originally false. Changing it to true still seems to work.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 

# -------------------------------------------------------------------------------------------------
## Build the DNN model
def build_model(parsed_args, device):
    # Initialize the neural network model
    if parsed_args.fair == False:
        g_net = GeneratorUnet1_1(in_chans=parsed_args.in_chans, out_chans=1, chans=parsed_args.chans, normalization = parsed_args.normalization)
    elif parsed_args.fair == True:
        g_net = GeneratorUnet1_1_FAIR(in_chans=parsed_args.in_chans, out_chans=1, chans=parsed_args.chans, normalization = parsed_args.normalization)
    
    if not parsed_args.no_parallel:
        # Set it to parallelize across visible GPUs
        g_net = torch.nn.DataParallel(g_net)
    
    # Move the model to the GPUs
    g_net.to(device)
    
    return g_net

# -------------------------------------------------------------------------------------------------
## Load the DNN model from disk
def load_model(parsed_args, device):
    checkpoint = torch.load(parsed_args.checkpoint)
    print(f"Generator Model loaded from {parsed_args.checkpoint}")
    
    parsed_args = checkpoint['parsed_args']
    g_net = build_model(parsed_args, device)
    g_net.load_state_dict(checkpoint['g_net'])

    optimizer_G = build_optim(parsed_args, g_net)
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    return checkpoint, g_net, optimizer_G

# -------------------------------------------------------------------------------------------------
## Build the optimizer
def build_optim(parsed_args, g_net):
    # Specify the optimizer's parameters - this is the optimzation algorithm that trains the  neural network
    optimizer_G = torch.optim.Adam(g_net.parameters(), lr=parsed_args.adam_lr, weight_decay=0)
    return optimizer_G

# -------------------------------------------------------------------------------------------------
## Initialize the loss criteria
def initialize_loss_criterion(params):
    # Options for possible loss functions
    loss_dict = {'dscloss':DiceCoeffLoss(), 'l1loss': torch.nn.L1Loss(), 'l2loss': torch.nn.MSELoss(), 'bceloss': torch.nn.BCELoss()}
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    # So use it in general when possible...
    
    # Set loss functionsn to use for training DNN
    criterion_g = loss_dict[params['criterion_g']]
    return criterion_g

# -------------------------------------------------------------------------------------------------
## Save the model to diisk
def save_model(parsed_args, params, epoch, g_net, optimizer_G, val_points, best_val_points, is_new_best):
    # NOTE: Save model every 10 epochs, otherwise it's excessive
    if epoch % params['checkpoint_interval'] == 0:
        torch.save(
            {
                'epoch': epoch,
                'parsed_args': parsed_args,
                'g_net': g_net.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'val_points': val_points,
                'best_val_points': best_val_points,
                'checkpoints_dir': params['checkpoints_dir']
            },
            # f = exp_dir / 'model_epoch{0:0>3d}.pt'.format(epoch)
            f = os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch))
        )
    if is_new_best:
        # This is the old code when every single epoch was being saved. Here, we can just copy it.
        # # shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        # shutil.copyfile(os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch)), 
        #    os.path.join(params['checkpoints_dir'], 'best_model.pt'.format(epoch)))
        # This is the new code when it's not guaranteed to have been saved, so resave it.
        torch.save(
            {
                'epoch': epoch,
                'parsed_args': parsed_args,
                'g_net': g_net.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'val_points': val_points,
                'best_val_points': best_val_points,
                'checkpoints_dir': params['checkpoints_dir']
            },
            # f = exp_dir / 'model_epoch{0:0>3d}.pt'.format(epoch)
            f = os.path.join(params['checkpoints_dir'], 'best_model.pt')
        )        

# -------------------------------------------------------------------------------------------------
# # Section V - The Main driver function
# -------------------------------------------------------------------------------------------------

def main(args):
    print('-'*80)
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print('-'*80)

    print('-'*80)
    print('Parsing arguments...')
    print('-'*80)
    parsed_args = parse_args(args)

    # Specify the device to run the code on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('-'*80)
    print('Seeding the pseudo-random number generators...')
    print('-'*80)    
    seed_prng(parsed_args, device) 

    print('-'*80)
    print('Initializing neural network model...')
    print('-'*80)

    if parsed_args.resume:
        checkpoint, g_net, optimizer_G = load_model(parsed_args, device)
        parsed_args = checkpoint['parsed_args']
        best_val_points = checkpoint['best_val_points']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        g_net = build_model(parsed_args, device)
        optimizer_G = build_optim(parsed_args, g_net)
        # best_val_loss = 1e9
        best_val_points = 0 # NOTE: if using PSNR, want it set low
        start_epoch = 0    

    logging.info(parsed_args)
    logging.info(g_net)
    
    # Create a dict out of the parsed_args
    params = create_params_dict(parsed_args, device)

    train_loader, val_loader, test_loader, display_loader, dataset_deets, req_filenames_dict = create_data_loaders(params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, params['lr_step_size'], params['lr_gamma'])
    # NOTE: There exist other interesting schedulers like ReduceLROnPlateau that are worth checking out
    criterion_g = initialize_loss_criterion(params)

    # args.exp_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir = args.exp_dir / 'summary') # Cool way to use '/' to combine two dirs
    writer = SummaryWriter(log_dir = os.path.join(params['checkpoints_dir'], 'summary'))

    try:
        print(f'''
        Starting training:
            Epochs: {params['epochs']}
            Batch size: {params['batch_size']}
            Dataset: {params['dataset']}
            Optimization Algorithm: 'Adam'
            Learning rate: {params['adam_lr']}
            Checkpoints: {str(params['checkpoints_dir'])}
            Device: {str(params['device'])}
        ''')

        tot_epochs = params['epochs']
        for epoch in range(start_epoch, params['epochs']):

            # Do the train step
            train_loss, train_time = train_epoch(params, epoch, g_net, criterion_g, train_loader, optimizer_G, scheduler, writer)

            # Do the validation/test step
            val_points, val_time = evaluate(params, dataset_deets, epoch, g_net, val_loader, writer, 'val', req_filenames_dict['val'])
            test_points, test_time = evaluate(params, dataset_deets, epoch, g_net, test_loader, writer, 'test', req_filenames_dict['test'])

            visualize_neurons(params, epoch, g_net, display_loader, writer)
            visualize(params, epoch, g_net, display_loader, writer)
            # is_new_best = val_loss < best_val_loss  # NOTE: If using PSNR, we actually want the highest loss lol
            is_new_best = val_points > best_val_points  # NOTE: If using PSNR, we actually want the highest loss lol
            best_val_points = max(best_val_points, val_points)
            save_model(parsed_args, params, epoch, g_net, optimizer_G, val_points, best_val_points, is_new_best)
            logging.info(
                f'Epoch = [{epoch+1:4d}/{tot_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValPoints = {val_points:.4g} TestPoints = {test_points:.4g}'
                f' TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s TestTime = {test_time:.4f}s',
            )
        # Write network graph to file
        # NOTE: 1) Doesn't work with dataparallel 2) Need to use the FAIR UNet code
        # sample = next(iter(display_loader))
        # writer.add_graph(g_net, sample['input_data'].to(device))                

        writer.close()
    
        # Free up memory used
        del g_net
        torch.cuda.empty_cache()        

    except KeyboardInterrupt:
        print('Code Interrupted!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)            

if __name__ == '__main__':
    main(sys.argv[1:])

# Include MS-SSIM
# Step 9 - interestingly it hallucinates the beads...
# Saving reconstructions to disk as h5 files from fastMRI is useful...
# tensorboard --samples_per_plugin images=100

# 1) https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# 2) A nice toolbox with a bunch of visualization techniques - https://github.com/utkuozbulak/pytorch-cnn-visualizations