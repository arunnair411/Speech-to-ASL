# To test UNet on the time domain missing packets
# k5s2
# For test on synthetic no_reverb - CUDA_VISIBLE_DEVICES=0 python test_v3.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test --data-split=val --store-dir=20200618_cleanvoip16timek5s2_no_reverb_denoising  --voip=cleanvoip --architecture=unet1dk5s2  --save-test-val-results --checkpoint=checkpoints/20200618_cleanvoip16timek5s2/best_model.pt  --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=4
# k15s4
# For test on synthetic no_reverb - CUDA_VISIBLE_DEVICES=0 python test_v3.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test --data-split=val --store-dir=20200618_cleanvoip16timek15s4_no_reverb_denoising --voip=cleanvoip --architecture=unet1dk15s4 --save-test-val-results --checkpoint=checkpoints/20200618_cleanvoip16timek15s4/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256
# Just checking if test_v3.py works after I made the apex amp additions - using uros's model as a guineapig
# CUDA_VISIBLE_DEVICES=1 python test_v3.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded --data-split=val --store-dir=20200624_DNS-challenge-expanded_onesecond_unetseparableuros_finprec        --voip=none      --architecture=unetseparable_uros --save-test-val-results --checkpoint=checkpoints/20200624_DNS-challenge-expanded_onesecond_unetseparableuros_finprec/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256

## 2020-07-14
# Testing on telecom distortions
# CUDA_VISIBLE_DEVICES=3 python test_v3.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded-teled --data-split=val --store-dir=20200714_RESULTS_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_telecomD --voip=cleanteled --architecture=unetseparable_uros --save-test-val-results --checkpoint=checkpoints/20200709_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_telecomD/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256
# CUDA_VISIBLE_DEVICES=1 python test_v3.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded --data-split=val --store-dir=20200714_RESULTS_DNS-challenge-expanded_gapfilling_small --voip=cleanvoip_v2 --architecture=unetseparable_uros_small --save-test-val-results --checkpoint=checkpoints/20200709_DNS-challenge-expanded_gapfilling_small/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256

## 2020-07-22
# CUDA_VISIBLE_DEVICES=0 python test_v4.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded-cleanvoip --pass-mask --average-init --data-split=val --store-dir=20200721_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleangapfillingTFwmaskAverageinit_TESTING --architecture=unetseparable_uros --save-test-val-results --checkpoint=checkpoints/20200721_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleangapfillingTFwmaskAverageinit/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256

## 2020-07-24
# TelecomD best performing network - TF domain UrosUNet
# CUDA_VISIBLE_DEVICES=0 python test_v4.py --gpu-ids 0 --dataset=DNS-challenge-synthetic-test-expanded-cleanteledstored --data-split=val --store-dir=20200722_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleantelecomDTF_TESTING --architecture=unetseparable_uros --save-test-val-results --checkpoint=checkpoints/20200722_DNS-challenge-expanded_onesecond_unetseparableuros_LSDloss_cleantelecomDTF/best_model.pt --no-visualization --no-neuron-visualization --no-model-copy --test-batch-size=256

import pdb
import os, sys, glob
import argparse, pathlib
import time
import shutil

# PyTorch DNN imports
import torch
import torchvision 

# U-Net related imports
# from unet import GeneratorUnet1_1, GeneratorUnet1_1_FAIR, UNetSeparable_64_uros, UNetSeparable_64_uros_small, UNetSeparable_64_uros_small_5, UNetSeparable_64, UNetSeparable_16, visualize_neurons, UNet1Dk5s2, UNet1Dk5s2_siren, UNet1Dk15s4
from models import PoseInterpolatorFC, PoseInterpolatorCNN

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
from utils import PoseInterpolatorFCDataset, PoseInterpolatorCNNDataset, ToTensor

# Validation/Testing data related inputs
# from utils import eval_net
from utils import eval_net_pose_interpolator

# Import NVIDIA AMP for mixed precision arithmetic
try:
    # sys.path.append('/home/t-arnair/Programs/apex')
    from apex import amp
    APEX_AVAILABLE = True
    # OPT_LEVEL = "O2"
    OPT_LEVEL = "O1"
except ModuleNotFoundError:
    APEX_AVAILABLE = False
# TO OVERRIDE AND JUST IGNORE APEX if necessary
# APEX_AVAILABLE = False

# NOTE: To set GPU Visibility from inside the code
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# NOTE: Store time-domain and time-frequency domain network architecture names in lists for easy clustering
time_frequency_domain_architectures = ['og','fair','unetseparable','unetseparable_uros', 'unetseparable_uros_small', 'unetseparable_uros_small_5']
time_domain_architectures = ['unet1dk15s4', 'unet1dk5s2', 'unet1dk5s2_siren']

# -------------------------------------------------------------------------------------------------
# # Section I - evaluate and visualize functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Testing function from fastMRI code
# def run_unet(args, model, data_loader):
#     model.eval()
#     reconstructions = defaultdict(list)
#     with torch.no_grad():
#         for (input, mean, std, fnames, slices) in data_loader:
#             input = input.unsqueeze(1).to(args.device)
#             recons = model(input).to('cpu').squeeze(1)
#             for i in range(recons.shape[0]):
#                 recons[i] = recons[i] * std[i] + mean[i]
#                 reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

#     reconstructions = {
#         fname: np.stack([pred for _, pred in sorted(slice_preds)])
#         for fname, slice_preds in reconstructions.items()
#     }
#     return reconstructions
#
# After testing using run_unet above, the reconstructions are written to file using save_reconstructions below
# def save_reconstructions(reconstructions, out_dir):
#     """
#     Saves the reconstructions from a model into h5 files that is appropriate for submission
#     to the leaderboard.

#     Args:
#         reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
#             corresponding reconstructions (of shape num_slices x height x width).
#         out_dir (pathlib.Path): Path to the output directory where the reconstructions
#             should be saved.
#     """
#     out_dir.mkdir(exist_ok=True)
#     for fname, recons in reconstructions.items():
#         with h5py.File(out_dir / fname, 'w') as f:
#             f.create_dataset('reconstruction', data=recons)

# -------------------------------------------------------------------------------------------------
## Evaluation function for val/test data
def evaluate(params, dataset_deets, epoch, g_net, criterion_g, data_loader, writer, data_string, req_filenames_clean, req_filenames_noisy):
    print('-'*80)
    print('Set the neural network to testing mode...')
    print('-'*80)
    # Set the network to eval mode to freeze BatchNorm weights
    g_net.eval()
    start = time.perf_counter()
    _PESQ_donothing, _PESQ,  _PSNR, _maeloss, _mseloss, _rmseloss, _lsdloss, _loss  = eval_net(g_net, criterion_g, req_filenames_clean, req_filenames_noisy, params, dataset_deets, data_string, data_loader)
    
    print(f"{data_string} DoNothing PESQ Score (Higher is better): {_PESQ_donothing}")
    print(f"{data_string} PESQ Score (Higher is better): {_PESQ}")
    print(f"{data_string} PSNR Score (Higher is better): {_PSNR}")
    print(f"{data_string} MAE Score (Lower is better): {_maeloss}")
    print(f"{data_string} MSE Score (Lower is better): {_mseloss}")
    print(f"{data_string} RMSE Score (Lower is better): {_rmseloss}")
    print(f"{data_string} LSD Score (Lower is better): {_lsdloss}")    
    print(f"{data_string} {params['criterion_g']} Loss (Lower is better): {_loss}")

    writer.add_scalar(f"Loss/{data_string}_pesq_donothing",  _PESQ_donothing, epoch)
    writer.add_scalar(f"Loss/{data_string}_pesq",  _PESQ, epoch)
    writer.add_scalar(f"Loss/{data_string}_MAE",  _maeloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_MSE",  _mseloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_RMSE",  _rmseloss, epoch)
    writer.add_scalar(f"Loss/{data_string}_LSD",  _lsdloss, epoch)
    # writer.add_scalar(f"Loss/{data_string}_psnr", _PSNR, epoch)
    writer.add_scalar(f"Loss/{data_string}_loss", _loss, epoch)

    mean_points = _PESQ
    mean_loss = _loss
    # return np.mean(losses), time.perf_counter() - start
    return _PESQ_donothing, mean_points, mean_loss, time.perf_counter() - start

# -------------------------------------------------------------------------------------------------
## Visualization function for val data - outputs it using a tensorboard writer to visually track progress
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

            save_image(target_output, f"Images/Target_{params['data_split']}", nrow=1)
            save_image(preds, f"Images/Reconstruction_{params['data_split']}", nrow=1)
            save_image(torch.abs(target_output - preds), f"Images/Error_{params['data_split']}", nrow=1)
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
    split_clean_ = []
    split_noisy_ = []
    for clean_data_path, noisy_data_path in zip(dataset_deets['clean_'+params['data_split']+'_data_path_list'], dataset_deets['noisy_'+params['data_split']+'_data_path_list']):
        curr_data_filenames_clean = sorted(glob.glob(os.path.join(clean_data_path,'*.wav')))
        curr_data_filenames_noisy = glob.glob(os.path.join(noisy_data_path,'*.wav'))
        curr_data_filenames_noisy.sort(key = lambda x: x.split('fileid')[1])
        split_clean_.extend(curr_data_filenames_clean)
        split_noisy_.extend(curr_data_filenames_noisy)
    
    # If there is no loaded data to denoise
    if not split_noisy_:
        print(f"No data corresponding to chosen data-split of {params['data_split']}_noisy. Exiting.")
        sys.exit(0)

    apply_on_clean = True if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanvoip', 'DNS-challenge-synthetic-test-expanded-cleanteledonline']  else False
    center_gap = True if params['architecture'] == 'unetseparable_uros_small_5' else False
    _transforms_list =[]
    # First, decide if you want to include gap filling
    if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanvoip', 'DNS-challenge-synthetic-test-expanded-noisyvoip']:
        _transforms_list.append(ApplyPacketLoss(apply_on_clean=apply_on_clean, pass_mask=params['pass_mask'], average_init=params['average_init']))
    # Second, decide if you want to include telecom distortions generated online
    if params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanteledonline', 'DNS-challenge-synthetic-test-expanded-noisyteledonline']:
        _transforms_list.append(ApplyTelecomDistortions(apply_on_clean=apply_on_clean))
    if params['architecture'] in params['time_frequency_domain_architectures']:
        _transforms_list.append(ApplySTFT(is_training=False))
        if params['pass_mask']:
            _transforms_list.append(TFGapFillingMaskEstimation())
            _transforms_list.append(ConcatMaskToInput(domain='TF'))
    elif params['architecture'] in params['time_domain_architectures']:
        if params['pass_mask']:
            _transforms_list.append(ConcatMaskToInput(domain='T'))
    _transforms_list.append(ToTensor())

    chosen_dataset   = MyDataset(clean_paths=split_clean_, noisy_paths=split_noisy_, is_training=False,
                            transform=transforms.Compose(_transforms_list))

    req_filenames_dict = {f"{params['data_split']}_clean": split_clean_, f"{params['data_split']}_noisy": split_noisy_}
    return chosen_dataset, dataset_deets, req_filenames_dict

# -------------------------------------------------------------------------------------------------
## Create dataloaders to load data from my dataset objects
def create_data_loaders(params):
    # mask_func = None
    # if args.mask_kspace:
    #     mask_func = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    # data = SliceData(
    #     root=args.data_path / f'{args.challenge}_{args.data_split}',
    #     transform=DataTransform(args.resolution, args.challenge, mask_func),
    #     sample_rate=1.,
    #     challenge=args.challenge
    # )
    # data_loader = DataLoader(
    #     dataset=data,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    # return data_loader    
    chosen_dataset, dataset_deets, req_filenames_dict = create_datasets(params)
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)] 
    if params['no_visualization']:
        display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // 1)]
    else:
        # display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // 64)] 
        display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // 4)] 
    
    data_loader = DataLoader(dataset=chosen_dataset, batch_size=params['test_batch_size'],
                                num_workers= 10*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    display_loader = DataLoader(dataset=display_data, batch_size=4, num_workers=10*len(params['gpu_ids']), pin_memory=True)
    
    return data_loader, display_loader, dataset_deets, req_filenames_dict

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
    parser.add_argument('--pass-mask', action='store_true', default=False,
                        help='Relevant to gap filling scenarios - if set, passes mask of gaps as additional input channel')
    parser.add_argument('--average-init', action='store_true', default=False,
                        help='Relevant to gap filling scenarios - if set, approximates missing gaps with average of previous and next packets')
    parser.add_argument('--data-split', choices=['val', 'test'], required=True, type=str,
                        help='Which data partition to run on: "val" or "test".')
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in mat files')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to an existing checkpoint. Required for testing.')
    parser.add_argument('--architecture', choices=['og', 'fair', 'unetseparable_uros', 'unetseparable_uros_small', 'unetseparable', 'unet1dk5s2', 'unet1dk5s2_siren', 'unet1dk15s4'], default='og', type=str,
                        help='og|fair|unetseparable_uros|unetseparable_uros_small|unetseparable|unet1dk5s2|unet1dk5s2_siren|unet1dk15s4')
    ## Parameters to update less often
    parser.add_argument('--no-visualization', action='store_true', default=False,
                        help='Disables visualization of the outputs; Also adjusts display_data step size to prevent errors')
    parser.add_argument('--no-neuron-visualization', action='store_true', default=False,
                        help='Disables visualization of the neurons ')
    parser.add_argument('--no-model-copy', action='store_true', default=False,
                        help='Disables copying the model to the results directory')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1000)')
    # parser.add_argument('--test-rows', type=int, default=128, metavar='R',
    #                     help='Number of rows in the test data inputs (default: 128)')
    # parser.add_argument('--test-cols', type=int, default=128, metavar='C',
    #                     help='Number of cols in the test data inputs (default: 128)')
    parser.add_argument('--batch-testing-id', type=int, default=0, 
                        help='Change output format to make it easier to do evaluation when using a bash script to do testing') # NOTE: ONLY CONSIDERED DURING TESTING - Default is 0, otherwise make it 1,2 etc from a bash file doing batch testing
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']                   = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']                   = parsed_args.dataset                     # Which dataset to processs
    params['pass_mask']         = parsed_args.pass_mask                   # Whether to pass a mask as well or not
    if params['pass_mask']:
        assert (params['dataset'] in ['DNS-challenge-synthetic-test-expanded-cleanvoip', 'DNS-challenge-synthetic-test-expanded-noisyvoip'] if params['pass_mask']==True else False), "pass_mask set to true but dataset is not set to gap filling"
    params['average_init']      = parsed_args.average_init                # Whether to initialize missing packet region with average of previous and next regions
    params['data_split']                = parsed_args.data_split                  # Which data partition to run on: "val" or "test"
    params['results_dir']               = os.path.join("results",parsed_args.store_dir)       # Directory in which to store the output images as mat files
    params['save_test_val_results']     = parsed_args.save_test_val_results       # Whether to save val and test outputs as .mat files
    params['checkpoint']                = parsed_args.checkpoint                  # Directory from which the model is being loaded if its being loaded; false otherwise
    params['architecture']              = parsed_args.architecture                # Set to 'og', 'fair', 'unetseparable_uros', 'unetseparable', 'unet1dk5s2', or 'unet1dk15s4' - determines the neural network architecture implementation
    params['no_visualization']          = parsed_args.no_visualization            # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_neuron_visualization']   = parsed_args.no_neuron_visualization     # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_model_copy']             = parsed_args.no_model_copy               # Whether to disable copying the model to the results directory    
    params['test_batch_size']           = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    # params['test_rows']                 = parsed_args.test_rows                   # Number of rows in the test images
    # params['test_cols']                 = parsed_args.test_cols                   # Number of columns in the test images
    params['batch_testing_id']          = parsed_args.batch_testing_id            # Whether to change output file format to make testing more amenable
    params['device']                    = device                                  # Device to run the code on

    params['time_frequency_domain_architectures'] = time_frequency_domain_architectures # List of time-frequency domain DNN architectures, specified at top of the file
    params['time_domain_architectures']           = time_domain_architectures           # List of time domain DNN architectures, specified at top of the file

    return params

# -------------------------------------------------------------------------------------------------
# # Section IV - Smaller Utility Functions
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
## Build the DNN model
def build_model(parsed_args, device):
    # Initialize the neural network model
    if parsed_args.architecture == 'og':
        g_net = GeneratorUnet1_1(in_chans=parsed_args.in_chans, out_chans=1, chans=parsed_args.chans, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'fair':
        g_net = GeneratorUnet1_1_FAIR(in_chans=parsed_args.in_chans, out_chans=1, chans=parsed_args.chans, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'unetseparable_uros' and parsed_args.data_loader_size == 'onesecond':
        g_net = UNetSeparable_64_uros(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)        
    elif parsed_args.architecture == 'unetseparable' and parsed_args.data_loader_size == 'onesecond':
        g_net = UNetSeparable_64(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)        
    elif parsed_args.architecture == 'unetseparable' and parsed_args.data_loader_size == 'quartersecond':
        g_net = UNetSeparable_16(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'unet1dk5s2' and parsed_args.data_loader_size == 'onesecond':
        g_net = UNet1Dk5s2(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'unet1dk5s2_siren' and parsed_args.data_loader_size == 'onesecond':
        g_net = UNet1Dk5s2_siren(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization, special_siren_init=False) # Works better
        # g_net = UNet1Dk5s2_siren(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization, special_siren_init=True)
    elif parsed_args.architecture == 'unet1dk15s4' and parsed_args.data_loader_size == 'onesecond':
        g_net = UNet1Dk15s4(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'unetseparable_uros_small' and parsed_args.data_loader_size == '8frames':
        g_net = UNetSeparable_64_uros_small(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)
    elif parsed_args.architecture == 'unetseparable_uros_small_5' and parsed_args.data_loader_size == '5frames':
        g_net = UNetSeparable_64_uros_small_5(in_chans=parsed_args.in_chans, out_chans=1, normalization = parsed_args.normalization)
    else:
        print('Unacceptable input arguments when building the network')
        sys.exit(0)
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
    
    parsed_args_model = checkpoint['parsed_args']
    g_net = build_model(parsed_args_model, device)
    g_net.load_state_dict(checkpoint['g_net'])

    return checkpoint, g_net, parsed_args_model.criterion_g, parsed_args_model.in_chans, parsed_args_model.normalization, parsed_args_model.no_parallel, parsed_args_model.data_loader_size

# -------------------------------------------------------------------------------------------------
## Initialize the loss criteria
def initialize_loss_criterion(params):
    # Options for possible loss functions
    loss_dict = {'dscloss':DiceCoeffLoss(), 'maeloss': torch.nn.L1Loss(), 'mseloss': torch.nn.MSELoss(), 'rmseloss': RMSELoss(), 'lsdloss': LSDLoss(), 'bceloss': torch.nn.BCELoss()}
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    # So use it in general when possible...
    
    # Set loss functionsn to use for training DNN
    criterion_g = loss_dict[params['criterion_g']]
    # if params['criterion_g']=='l1loss':
    #     criterion_g = loss_dict['maeloss']
    # else:    
    #     criterion_g = loss_dict[params['criterion_g']]
    return criterion_g

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

    # Create a dict out of the parsed_args
    params = create_params_dict(parsed_args, device)

    print('-'*80)
    print('Create the data loaders...')
    print('-'*80)
    data_loader, display_loader, dataset_deets, req_filenames_dict = create_data_loaders(params)

    print('-'*80)
    print('Load the model from disk...')
    print('-'*80)
    checkpoint, g_net, params['criterion_g'], params['in_chans'], params['normalization'], params['no_parallel'], params['data_loader_size'] = load_model(parsed_args, device)

    model_epoch = checkpoint['epoch']
    del checkpoint

    criterion_g = initialize_loss_criterion(params)
    writer = SummaryWriter(log_dir = os.path.join(params['results_dir'], 'summary'))

    # Do the validation/test step
    do_nothing_points, _points, _loss, _time = evaluate(params, dataset_deets, model_epoch, g_net, criterion_g, data_loader, writer, params['data_split'], req_filenames_dict[f"{params['data_split']}_clean"], req_filenames_dict[f"{params['data_split']}_noisy"])
    
    if not params['no_neuron_visualization']:
        visualize_neurons(params, model_epoch, g_net, display_loader, writer)
    if not params['no_visualization']:
        visualize(params, model_epoch, g_net, display_loader, writer)

    os.makedirs(os.path.join(params['results_dir'],'model'), exist_ok=True)
    if not params['no_model_copy']:
        # Copy the run model file into the results directory        
        shutil.copyfile(parsed_args.checkpoint, os.path.join(os.path.join(params['results_dir'],'model','copied_model.pt')))    

    # Save the test argparse into the results directory
    save_file_string  = os.path.join(params['results_dir'], 'testing_deets.txt')
    with open(save_file_string, 'w+') as f: # we don't have to write "file.close()". That will automatically be called.
        for key, value in params.items():
            f.write(f"{key}:{value}\n")

    # Write network graph to file
    # NOTE: 1) Doesn't work with dataparallel 2) Need to use the FAIR UNet code
    # sample = next(iter(display_loader))
    # writer.add_graph(g_net, sample['input_data'].to(device))

    writer.close()
    del g_net
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main(sys.argv[1:])
