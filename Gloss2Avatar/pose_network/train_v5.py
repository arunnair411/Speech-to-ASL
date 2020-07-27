# Expt8: Training unet1dk5s2 with MAELoss+T telecomdistortions - running alongside experiment 3
# CUDA_VISIBLE_DEVICES=2 python train_v4.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded-cleanteledstored --data-loader-size=onesecond --data-loader-style=dns --store-dir=20200722_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleantelecomDT  --architecture=unet1dk5s2 --in-chans=1 --normalization=batchnorm --criterion-g=maeloss --epochs=500 --batch-size=64 --test-batch-size=256 --adam-lr=1e-4 --adam-beta1=0.5 --adam-beta2=0.9 --lr-step-size=166 --lr-gamma=0.5 --teststats-save-interval=2 --checkpoint-interval=30
# Resuming training
# CUDA_VISIBLE_DEVICES=2 python train_v4.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded-cleanteledstored --data-loader-size=onesecond --data-loader-style=dns --store-dir=20200722_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleantelecomDT  --resume --checkpoint=checkpoints/20200722_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleantelecomDT/best_model.pt --architecture=unet1dk5s2 --in-chans=1 --normalization=batchnorm --criterion-g=maeloss --epochs=500 --batch-size=64 --test-batch-size=256 --adam-lr=1e-4 --adam-beta1=0.5 --adam-beta2=0.9 --lr-step-size=166 --lr-gamma=0.5 --teststats-save-interval=2 --checkpoint-interval=30
##############################################################################################################################################

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
# from unet import GeneratorUnet1_1, GeneratorUnet1_1_FAIR, UNetSeparable_64_uros, UNetSeparable_64_uros_small, UNetSeparable_64_uros_small_5, UNetSeparable_64, UNetSeparable_16, visualize_neurons, UNet1Dk5s2, UNet1Dk5s2_siren, UNet1Dk15s4
from models import PosePredictorFC, PosePredictorCNN

# Loss Function Imports
from utils import DiceCoeffLoss, RMSELoss, LSDLoss

# Tensorboard import
from torch.utils.tensorboard import SummaryWriter

# Dataset imports
from utils import retrieve_dataset_filenames
import sklearn

# Dataloader imports
from torch.utils.data import DataLoader
from torchvision import transforms
# from utils import MyDataset, MyDataset_Kaz_Training_Raw, ApplySTFT, ApplyPacketLoss, ApplyTelecomDistortions, RandomSubsample, TFGapFillingMaskEstimation, ConcatMaskToInput, ToTensor
from utils import PosePredictorFCDataset, PosePredictorCNNDataset, ToTensor

# Validation/Testing data related inputs
# from utils import eval_net
from utils import eval_net_pose_predictor

# Import NVIDIA AMP for mixed precision arithmetic
try:
    # sys.path.append('/home/t-arnair/Programs/apex')
    from apex import amp
    APEX_AVAILABLE = True
    DROP_LAST = True # Need to drop the last batch or an out-of-memory error occurs
    # OPT_LEVEL = "O2"
    OPT_LEVEL = "O1"
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    DROP_LAST = False
# TO OVERRIDE AND JUST IGNORE APEX if necessary
# APEX_AVAILABLE = False
# DROP_LAST = False

# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: To set GPU Visibility from inside the code
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
        # Preprocess the data (if required)
        input_data = sample['input_data']
        target_output = sample['target_output']

        # Copy the data to GPU
        input_data = input_data.to(params['device'])
        target_output = target_output.to(params['device'])
        target_output_flat = target_output.view(-1) # Flatten (i.e.) vectorize the data

        # Pass the minibatch through the network to get predictions
        preds = g_net(input_data)
        # preds = torch.sigmoid(preds) # NOTE: Don't need the sigmoid... just removing it in favor of a plain conv like Dung did...
        preds_flat = preds.view(-1)

        # Calculate the loss for the batch
        # if params['criterion_g']=='lsdloss': # Here, can't use flattened data
        #     loss = criterion_g(preds, target_output)
        # else: # Otherwise, needs flattened data
        #     loss = criterion_g(preds_flat, target_output_flat)
        loss = criterion_g(preds_flat, target_output_flat)

        # Zero the parameter gradients
        optimizer_G.zero_grad()

        # Compute the gradients and take a step
        if APEX_AVAILABLE:
            with amp.scale_loss(loss, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer_G.step()
        
        g_epoch_loss += loss.item() # Note: calling .item() creates a a copy which is stored in g_epoch_loss... so modifying the latter doesn't modify the former. Verified!
        # avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item() - FAIR - dunno what is going on here...
        writer.add_scalar(f"Train/TrainLoss", loss.item(), global_step + iterIdx)

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

    writer.add_scalar(f"Train/TrainLoss_EpochAvg", avg_loss, epoch)

    return avg_loss, time.perf_counter() - start_epoch

# -------------------------------------------------------------------------------------------------
## Evaluation function for val/test data
def evaluate(params, dataset_deets, epoch, g_net, criterion_g, data_loader, writer, data_string, req_filenames):
    print('-'*80)
    print('Set the neural network to testing mode...')
    print('-'*80)
    # Set the network to eval mode to freeze BatchNorm weights
    g_net.eval()
    start = time.perf_counter()
    _maeloss, _mseloss, _rmseloss, _loss  = eval_net_pose_predictor(g_net, criterion_g, req_filenames, params, dataset_deets, data_string, data_loader)

    print(f"{data_string} MAE Score (Lower is better): {_maeloss}")
    print(f"{data_string} MSE Score (Lower is better): {_mseloss}")
    print(f"{data_string} RMSE Score (Lower is better): {_rmseloss}")
    print(f"{data_string} {params['criterion_g']} Loss (Lower is better): {_loss}")

    writer.add_scalar(f"Loss/{data_string}_MAE",  _maeloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_MSE",  _mseloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_RMSE",  _rmseloss, epoch)
    # writer.add_scalar(f"Loss/{data_string}_psnr", _PSNR, epoch)
    writer.add_scalar(f"Loss/{data_string}_loss", _loss, epoch)

    # mean_points = _PESQ
    mean_points = -1*_loss
    mean_loss = _loss
    # return np.mean(losses), time.perf_counter() - start
    return mean_points, mean_loss, time.perf_counter() - start

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
            # preds = torch.sigmoid(preds) # No sigmoid!

            save_image(target_output, 'Images/Target', nrow=1)
            save_image(preds, 'Images/Reconstruction', nrow=1)
            save_image(torch.abs(target_output - preds), 'Images/Error', nrow=1)
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

    training_datasets = ['pose_npy_dir', 'npy_dir']

    print('-'*30)
    print('Reading in data filenames...')
    print('-'*30)

    if dataset_deets['dataset'] in training_datasets:
        # Read in data filenames and split into train-val-test
        split_ = []
        for data_path in dataset_deets['data_path_list']:
            split_.extend(sorted(glob.glob(os.path.join(data_path,'*.npy'))))
        split_train, split_val_test = sklearn.model_selection.train_test_split(split_, test_size=0.2, random_state=1337)
        split_val, split_test = sklearn.model_selection.train_test_split(split_val_test, test_size=0.5, random_state=1337)
    else: 
        # Not a training dataset
        print('Trying to train on a test set. Exiting.')
        sys.exit(0)
    if params['trunc_data_flag']:
        # Truncate the dataset to be more manageable - use this to check if neural network code is right
        # It should overfit to the reduced size data and give near 100% results...
        split_train = split_train[0:64]
        split_val   = split_val[0:64]
        split_test  = split_test[0:64]

    train_transforms_list = []
    val_test_transforms_list =[]
    
    # First, if it is the FC pose prediction network
    if params['dataset'] in ['pose_npy_dir'] and params['architecture'] in ['PosePredictorFC']:
        ## TRANSFORMS
        # edge_sample_length determines the number of known pose samples at each end, sample_gap is the number of pose frames to interpolate
        train_transforms_list.append(PoseSubsampler(edge_sample_length=1, sample_gap =10))
        # Don't need val_test_transforms_list.append(...) as we are operating on the entire data
        train_transforms_list.append(ToTensor())
        val_test_transforms_list.append(ToTensor())

        ## DATASET
        train_data = PosePredictorFCDataset(file_paths=split_train,
                                transform=transforms.Compose(train_transforms_list))
        val_data   = PosePredictorFCDataset(file_paths=split_val,
                                transform=transforms.Compose(val_test_transforms_list))
        test_data  = PosePredictorFCDataset(file_paths=split_test,
                                transform=transforms.Compose(val_test_transforms_list))
    # Second, if it is the CNN pose prediction network
    elif params['dataset'] in ['pose_npy_dir'] and params['architecture'] in ['PosePredictorCNN']:
        ## TRANSFORMS
        train_transforms_list.append(PoseSubsampler(edge_sample_length=1, sample_gap =10))
        train_transforms_list.append(PointToHeatmapConverter())
        val_test_transforms_list.append(PointToHeatmapConverter())
        train_transforms_list.append(ToTensor())
        val_test_transforms_list.append(ToTensor())

        ## DATASET
        train_data = PosePredictorCNNDataset(file_paths=split_train,
                                transform=transforms.Compose(train_transforms_list))
        val_data   = PosePredictorCNNDataset(file_paths=split_val,
                                transform=transforms.Compose(val_test_transforms_list))
        test_data  = PosePredictorCNNDataset(file_paths=split_test,
                                transform=transforms.Compose(val_test_transforms_list))        
    req_filenames_dict = {'train': split_train, 'val': split_val, 'test': split_test}
    return train_data, val_data, test_data, dataset_deets, req_filenames_dict

# -------------------------------------------------------------------------------------------------
## Create dataloaders to load data from my dataset objects
def create_data_loaders(params):
    train_data, val_data, test_data, dataset_deets, req_filenames_dict = create_datasets(params)
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)] 
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 64)] 
    display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 4)] 

    train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'],
                                shuffle=True, num_workers= 10*len(params['gpu_ids']), pin_memory=True, drop_last=DROP_LAST) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    val_loader   = DataLoader(dataset=val_data, batch_size=params['test_batch_size'],
                                num_workers= 10*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    test_loader  = DataLoader(dataset=test_data, batch_size=params['test_batch_size'],
                                num_workers= 10*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
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

    # display_loader = DataLoader(dataset=display_data, batch_size=64, num_workers=2*len(params['gpu_ids']), pin_memory=True)
    display_loader = DataLoader(dataset=display_data, batch_size=4, num_workers=2*len(params['gpu_ids']), pin_memory=True)
    
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
    # IMPORTANT: usage --dataset=DNS-challenge-synthetic-test
    parser.add_argument('--dataset', required=True, type=str,
                        help='<predefined dataset name>|<path to dataset>')
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in files')
    ## Parameters to update less often
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')    
    parser.add_argument('--checkpoint', type=str, default=False,
                        help='Path to an existing checkpoint. Used along with "--resume"')    
    parser.add_argument('--prng-seed', type=int, default=1337, metavar='S',
                        help='Seed for all the pseudo-random number generators')
    parser.add_argument('--architecture', choices=['PosePredictorFC', 'PosePredictorCNN'], default='PosePredictorFC', type=str,
                        help='PosePredictorFC|PosePredictorCNN')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help='Flag to prevent paralellization of the model across the GPUs')
    parser.add_argument('--in-chans', default=1, type=int, metavar='IC',
                        help='Number of input channels') # TODO - adapt this to set edge_sample_length and sample_gap
    # parser.add_argument('--normalization', choices=['none','batchnorm','instancenorm'], default='batchnorm', type=str,
    #                     help='none|batchnorm|instancenorm') # TODO - adapt this according to Oscar's code...
    parser.add_argument('--criterion-g', choices=['dscloss','mseloss', 'rmseloss', 'maeloss', 'bceloss'], required=True,
                        help='dscloss|mseloss|rmseloss|maeloss|bceloss')
    # IMPORTANT: usage --epochs=80 or --epochs 80 (both work)
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='Number of epochs to train (default: 80)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Input mini-batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1)') # need it to be 1, otherwise different sequences have different numbers of frames...
    parser.add_argument('--trunc-data-flag', action='store_true', default=False,
                        help='Work with truncated dataset')
    parser.add_argument('--adam-lr', type=float, default=0.0001, metavar='LR',
                        help='Adam learning rate (default: 1e-4)')
    parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='AB1',
                        help='Adam beta 1 term (default: 0.9)')
    parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='AB2',
                        help='Adam beta 2 term (default: 0.999)')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    # parser.add_argument('--flip-training', action='store_true', default=False,
    #                     help='Whether to flip the training data')
    # parser.add_argument('--test-rows', type=int, default=128, metavar='R',
    #                     help='Number of rows in the test data inputs (default: 128)')
    # parser.add_argument('--test-cols', type=int, default=128, metavar='C',
    #                     help='Number of cols in the test data inputs (default: 128)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many mini-batches to wait before logging training status')
    parser.add_argument('--teststats-save-interval', type=int, default=1, metavar='TSI',
                        help='How many epochs to wait before testing on val/test data and saving stats to tensorboard')
    parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='CI',
                        help='How many epochs to wait before saving a model file')
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']           = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']           = parsed_args.dataset                     # Which dataset to processs
    params['checkpoints_dir']   = os.path.join("checkpoints", parsed_args.store_dir)   # Directory in which to store the model checkpoints and training details as .pt files
    params['results_dir']       = os.path.join("results", parsed_args.store_dir)       # Directory in which to store the outputs
    params['save_test_val_results'] = parsed_args.save_test_val_results   # Whether to save val and test outputs
    params['resume']            = parsed_args.resume                      # Flag set to true if training is to be resumed from provided checkpoint
    params['checkpoint']        = parsed_args.checkpoint                  # Where to load the model from if it is being loaded; false otherwise
    params['architecture']      = parsed_args.architecture                # Set to 'og', 'fair', 'unetseparable_uros', 'unetseparable', 'unet1dk5s2', or 'unet1dk15s4' - determines the neural network architecture implementation
    params['no_parallel']       = parsed_args.no_parallel                 # Flag to prevent model paralellization across GPUs
    params['in_chans']          = parsed_args.in_chans                    # Number of input channels - defaults to 1
    params['normalization']     = parsed_args.normalization               # Set to 'none', 'batchnorm', or 'instancenorm'
    params['criterion_g']       = parsed_args.criterion_g                 # DNN loss function
    params['epochs']            = parsed_args.epochs                      # Total number of training epochs i.e. complete passes of the training data
    params['batch_size']        = parsed_args.batch_size                  # Number of training files in one mini-batch
    params['test_batch_size']   = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    params['trunc_data_flag']   = parsed_args.trunc_data_flag             # Flag on whether to truncate the dataset
    params['adam_lr']           = parsed_args.adam_lr                     # Learning rate for the Adam Optimzer
    params['adam_beta1']        = parsed_args.adam_beta1                  # Adam beta 1 term
    params['adam_beta2']        = parsed_args.adam_beta2                  # Adam beta 2 term
    params['lr_step_size']      = parsed_args.lr_step_size                # Period of learning rate decay - after this number of epochs, the lr will decrease by the multiplicative factor of lr_gamma
    params['lr_gamma']          = parsed_args.lr_gamma                    # Multiplicative factor of learning rate decay. Default: 0.1.    
    # params['to_flip']           = int(parsed_args.flip_training)          # Whether to augment the training data through flipping it laterally
    # params['test_rows']         = parsed_args.test_rows                   # Number of rows in the test images
    # params['test_cols']         = parsed_args.test_cols                   # Number of columns in the test images
    params['log_interval']              = parsed_args.log_interval              # How often to print loss information for a minibatch
    params['teststats_save_interval']   = parsed_args.teststats_save_interval   # How often to run test data and save test stats
    params['checkpoint_interval']       = parsed_args.checkpoint_interval       # How often to save a snapshot of the model
    params['device']            = device                                  # Device to run the code on

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
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True # Changing this to true from false 1) Speeds up the code a bit - 160s instead of 180s 2) Mostly preserves deterministic behavior (ran it thrice, runs 2 and 3 were identical, 1 was slightly different)
        torch.backends.cudnn.deterministic = True # Changing this to false for my use case didn't make a difference in speed

# -------------------------------------------------------------------------------------------------
## Build the DNN model
def build_model(parsed_args, device):
    # Initialize the neural network model
    if parsed_args.architecture == 'PosePredictorFC':
        # g_net = PosePredictorFC(in_chans=parsed_args.in_chans, out_chans=10, normalization = parsed_args.normalization) # TODO: Normalization?
        g_net = PosePredictorFC(in_chans=parsed_args.in_chans, out_chans=10)
    elif parsed_args.architecture == 'PosePredictorCNN':
        # g_net = PosePredictorCNN(in_chans=parsed_args.in_chans, out_chans=10, chans=parsed_args.chans, normalization = parsed_args.normalization) # TODO: Chans and Normalization?
        g_net = PosePredictorCNN(in_chans=parsed_args.in_chans, out_chans=10)
    else:
        print('Unacceptable input arguments when building the network')
        sys.exit(0)
    
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
    optimizer_G = build_optim(parsed_args, g_net)

    if APEX_AVAILABLE:
        if OPT_LEVEL != "O1":                    
            g_net, optimizer_G = amp.initialize(
                g_net, optimizer_G, opt_level=OPT_LEVEL,
                keep_batchnorm_fp32=True, loss_scale="dynamic"
                )
        else:
            g_net, optimizer_G = amp.initialize(
                g_net, optimizer_G, opt_level=OPT_LEVEL,
                keep_batchnorm_fp32=None, loss_scale="dynamic"
                ) # Change keep_batchnorm_fp32 to None otherwise it throws an error - bathnorm is kept as fp32 by default in "O1"
        amp.load_state_dict(checkpoint['amp'])    
    checkpoint['g_net'] = {key.replace('module.', ''):value for key, value in checkpoint['g_net'].items()} # Need to ``un-parallelize'' the dict before loading it... - because of the amp change
    g_net.load_state_dict(checkpoint['g_net'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    return checkpoint, g_net, optimizer_G

# -------------------------------------------------------------------------------------------------
## Build the optimizer
def build_optim(parsed_args, g_net):
    # Specify the optimizer's parameters - this is the optimzation algorithm that trains the  neural network
    optimizer_G = torch.optim.Adam(g_net.parameters(), lr=parsed_args.adam_lr, betas=(parsed_args.adam_beta1, parsed_args.adam_beta2), weight_decay=0)
    return optimizer_G

# -------------------------------------------------------------------------------------------------
## Initialize the loss criteria
def initialize_loss_criterion(params):
    # Options for possible loss functions
    loss_dict = {'dscloss':DiceCoeffLoss(), 'maeloss': torch.nn.L1Loss(), 'mseloss': torch.nn.MSELoss(), 'rmseloss': RMSELoss(), 'bceloss': torch.nn.BCELoss()}
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    # So use it in general when possible...
    
    # Set loss functionsn to use for training DNN
    criterion_g = loss_dict[params['criterion_g']]
    return criterion_g

# -------------------------------------------------------------------------------------------------
## Save the model to diisk
def save_model(parsed_args, params, epoch, g_net, optimizer_G, val_points, best_val_points, is_new_best):
    # NOTE: Save model every 10 epochs, otherwise it's excessive
    checkpoint= {
                    'epoch': epoch,
                    'parsed_args': parsed_args,
                    'g_net': g_net.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'val_points': val_points,
                    'best_val_points': best_val_points,
                    'checkpoints_dir': params['checkpoints_dir']
                }
    if APEX_AVAILABLE:
        checkpoint['amp'] = amp.state_dict()
    if epoch % params['checkpoint_interval'] == 0:
        torch.save(checkpoint,
                # f = exp_dir / 'model_epoch{0:0>3d}.pt'.format(epoch)
                f = os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch))
            )
    if is_new_best:
        # This is the old code when every single epoch was being saved. Here, we can just copy it.
        # # shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        # shutil.copyfile(os.path.join(params['checkpoints_dir'], 'model_epoch{0:0>3d}.pt'.format(epoch)), 
        #    os.path.join(params['checkpoints_dir'], 'best_model.pt'.format(epoch)))
        # This is the new code when it's not guaranteed to have been saved, so resave it.
        torch.save(checkpoint,
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
        new_num_epochs = parsed_args.epochs
        checkpoint, g_net, optimizer_G = load_model(parsed_args, device)
        parsed_args = checkpoint['parsed_args']
        parsed_args.epochs = new_num_epochs # Keep this as the new value
        parsed_args.resume = True # Don't want this overwritten...
        best_val_points = checkpoint['best_val_points']
        start_epoch = checkpoint['epoch']+1 # NOTE: Need the +1 since we don't want to run the saved epoch again
        del checkpoint
    else:
        g_net = build_model(parsed_args, device)
        optimizer_G = build_optim(parsed_args, g_net)
        # best_val_loss = 1e9
        best_val_points = 0 # NOTE: if using PSNR/PESQ, want it set low
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

        if APEX_AVAILABLE and not parsed_args.resume: # if parsed_args.resume, then amp is already initialized prior
            if OPT_LEVEL != "O1":
                g_net, optimizer_G = amp.initialize(
                    g_net, optimizer_G, opt_level=OPT_LEVEL,
                    keep_batchnorm_fp32=True, loss_scale="dynamic"
                    )
            else:
                g_net, optimizer_G = amp.initialize(
                    g_net, optimizer_G, opt_level=OPT_LEVEL,
                    keep_batchnorm_fp32=None, loss_scale="dynamic"
                    ) # Change keep_batchnorm_fp32 to None otherwise it throws an error - bathnorm is kept as fp32 by default in "O1"

        if not params['no_parallel']:
            # Set it to parallelize across visible GPUs - this needs to be after the call to amp.initialize
            g_net = torch.nn.DataParallel(g_net)

        for epoch in range(start_epoch, params['epochs']):

            # Do the train step
            train_loss, train_time = train_epoch(params, epoch, g_net, criterion_g, train_loader, optimizer_G, scheduler, writer)

            if (epoch % params['teststats_save_interval'] == 0) or (epoch % params['checkpoint_interval'] == 0):
                # Do the validation/test step
                val_points, val_loss, val_time     = evaluate(params, dataset_deets, epoch, g_net, criterion_g, val_loader,  writer, 'val',  req_filenames_dict['val'])
                test_points, test_loss, test_time = evaluate(params, dataset_deets, epoch, g_net, criterion_g, test_loader, writer, 'test', req_filenames_dict['test'])
                # visualize_neurons(params, epoch, g_net, display_loader, writer)
                # visualize(params, epoch, g_net, display_loader, writer)
                # is_new_best = val_loss < best_val_loss  # NOTE: If using PSNR/PESQ, we actually want the highest loss lol
                is_new_best = val_points > best_val_points  # NOTE: If using PSNR/PESQ, we actually want the highest loss lol
                best_val_points = max(best_val_points, val_points)
                save_model(parsed_args, params, epoch, g_net, optimizer_G, val_points, best_val_points, is_new_best)
                logging.info(
                    f'Epoch = [{epoch+1:4d}/{tot_epochs:4d}] TrainLoss = {train_loss:.4g} '
                    f'DoNothingValPoints = {do_nothing_val_points:.4g} DoNothingTestPoints = {do_nothing_test_points:.4g} '
                    f'ValPoints = {val_points:.4g} TestPoints = {test_points:.4g} '
                    f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s TestTime = {test_time:.4f}s',
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
# tensorboard --samples_per_plugin images=100