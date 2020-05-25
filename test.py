# # Sampling=0.05
# For val - CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 --dataset=20200102_Sim3_0.05 --data-split=val  --store-dir=20200114_20200102_Sim3_l1_TEST  --save-test-val-results --checkpoint=checkpoints/20200114_20200102_Sim3_l1/model_epoch199.pt --n-channels=2 --test-batch-size=1024 --test-rows=128 --test-cols=128
# For test- CUDA_VISIBLE_DEVICES=0 python test.py --gpu-ids 0 1 --dataset=20200102_Sim3_0.05 --data-split=test --store-dir=20200114_20200102_Sim3_l1_TEST  --save-test-val-results --checkpoint=checkpoints/20200114_20200102_Sim3_l1/model_epoch199.pt --n-channels=2 --test-batch-size=1024 --test-rows=128 --test-cols=128  

# 2020-01-21 - Visualization test
# # Sampling=0.05
# For val - CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --no-parallel --dataset=20200102_Sim3_0.05 --data-split=val  --store-dir=20200120_20200102_Sim3_l1_TEST  --save-test-val-results --checkpoint=checkpoints/20200120_20200102_Sim3_l1/best_model.pt --test-batch-size=256 --test-rows=128 --test-cols=128
# For test- CUDA_VISIBLE_DEVICES=1 python test.py --gpu-ids 0 --no-parallel --dataset=20200102_Sim3_0.05 --data-split=test --store-dir=20200120_20200102_Sim3_l1_TEST  --save-test-val-results --checkpoint=checkpoints/20200120_20200102_Sim3_l1/best_model.pt --test-batch-size=256 --test-rows=128 --test-cols=128

import os, sys, glob
import argparse, pathlib
import time
import shutil

# PyTorch DNN imports
import torch
import torchvision 

# U-Net related imports
from unet import GeneratorUnet1_1, GeneratorUnet1_1_FAIR, visualize_neurons

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

    mean_score = _PSNR
    # return np.mean(losses), time.perf_counter() - start
    return mean_score, time.perf_counter() - start

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
            preds = torch.sigmoid(preds)

            save_image(target_output, f"Images/Target_{params['data_split']}", nrow=8)
            save_image(preds, f"Images/Reconstruction_{params['data_split']}", nrow=8)
            save_image(torch.abs(target_output - preds), f"Images/Error_{params['data_split']}", nrow=8)
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
    split_ = []
    
    for data_path in dataset_deets[f"{params['data_split']}_data_path_list"]:
        curr_data_filenames = sorted(glob.glob(os.path.join(data_path,'*.mat')))
        split_.extend(curr_data_filenames)
    
    # If there is no loaded data
    if not split_:
        print(f"No data corresponding to chosen data-split of {params['data_split']}. Exiting.")
        sys.exit(0)

    chosen_dataset = MyDataset(mat_paths=split_,
                            transform=transforms.Compose([
                                RandomHorizontalFlipArun(0), 
                                ToTensor()
                            ]))

    req_filenames_dict = {params['data_split'] : split_}
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
        display_data = [chosen_dataset[i] for i in range(0, len(chosen_dataset), len(chosen_dataset) // 64)] 
    
    data_loader = DataLoader(dataset=chosen_dataset, batch_size=params['test_batch_size'],
                                num_workers= 2*len(params['gpu_ids']), pin_memory=True) # = 2*#GPUs as #GPUs didn't give 100% utilization for small files
    display_loader = DataLoader(dataset=display_data, batch_size=64, num_workers=2*len(params['gpu_ids']), pin_memory=True)
    
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
    # IMPORTANT: usage --dataset=anechoic+hyp
    parser.add_argument('--dataset', required=True, type=str,
                        help='<predefined dataset name>|<path to dataset>')
    parser.add_argument('--data-split', choices=['val', 'test'], required=True, type=str,
                        help='Which data partition to run on: "val" or "test".')
    # IMPORTANT: usage --store-dir=20190102_20200102_Sim1
    parser.add_argument('--store-dir', required=True, type=pathlib.Path,
                        help='Name of output directory')
    parser.add_argument('--save-test-val-results', action='store_true', default=False,
                        help='Whether to save val and test outputs in mat files')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to an existing checkpoint. Required for testing.')
    ## Parameters to update less often
    parser.add_argument('--no-visualization', action='store_true', default=False,
                        help='Disables visualization of the outputs; Also adjusts display_data step size to prevent errors')
    parser.add_argument('--no-neuron-visualization', action='store_true', default=False,
                        help='Disables visualization of the neurons ')
    parser.add_argument('--no-model-copy', action='store_true', default=False,
                        help='Disables copying the model to the results directory')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N_T',
                        help='Input mini-batch size for testing (default: 1000)')
    parser.add_argument('--test-rows', type=int, default=128, metavar='R',
                        help='Number of rows in the test data inputs (default: 128)')
    parser.add_argument('--test-cols', type=int, default=128, metavar='C',
                        help='Number of cols in the test data inputs (default: 128)')
    parser.add_argument('--batch-testing-id', type=int, default=0, 
                        help='Change output format to make it easier to do evaluation when using a bash script to do testing') # NOTE: ONLY CONSIDERED DURING TESTING - Default is 0, otherwise make it 1,2 etc from a bash file doing batch testing
    return parser.parse_args(args)

# -------------------------------------------------------------------------------------------------
## Convert parsed arguments to a dict for more easy manipulation
def create_params_dict(parsed_args, device):
    params = {}
    params['gpu_ids']                   = parsed_args.gpu_ids                     # ids of GPUs being used
    params['dataset']                   = parsed_args.dataset                     # Which dataset to processs
    params['data_split']                = parsed_args.data_split                  # Which data partition to run on: "val" or "test"
    params['results_dir']               = os.path.join("results",parsed_args.store_dir)       # Directory in which to store the output images as mat files
    params['save_test_val_results']     = parsed_args.save_test_val_results       # Whether to save val and test outputs as .mat files
    params['checkpoint']                = parsed_args.checkpoint                  # Directory from which the model is being loaded if its being loaded; false otherwise
    params['no_visualization']          = parsed_args.no_visualization            # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_neuron_visualization']   = parsed_args.no_neuron_visualization     # Whether to run visualization of the neurons and outputs; Also adjusts display_data step size to prevent errors
    params['no_model_copy']             = parsed_args.no_model_copy               # Whether to disable copying the model to the results directory    
    params['test_batch_size']           = parsed_args.test_batch_size             # Number of testing files in one mini-batch - can be much larger since gradient information isn't required
    params['test_rows']                 = parsed_args.test_rows                   # Number of rows in the test images
    params['test_cols']                 = parsed_args.test_cols                   # Number of columns in the test images
    params['batch_testing_id']          = parsed_args.batch_testing_id            # Whether to change output file format to make testing more amenable
    params['device']                    = device                                  # Device to run the code on
    
    return params

# -------------------------------------------------------------------------------------------------
# # Section IV - Smaller Utility Functions
# -------------------------------------------------------------------------------------------------

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
    
    parsed_args_model = checkpoint['parsed_args']
    g_net = build_model(parsed_args_model, device)
    g_net.load_state_dict(checkpoint['g_net'])

    return checkpoint, g_net, parsed_args_model.in_chans, parsed_args_model.fair, parsed_args_model.normalization, parsed_args_model.no_parallel


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
    checkpoint, g_net, params['in_chans'], params['fair'], params['normalization'], params['no_parallel'] = load_model(parsed_args, device)
    
    model_epoch = checkpoint['epoch']
    del checkpoint

    writer = SummaryWriter(log_dir = os.path.join(params['results_dir'], 'summary'))    

    # Do the validation/test step
    _score, _time = evaluate(params, dataset_deets, model_epoch, g_net, data_loader, writer, params['data_split'], req_filenames_dict[params['data_split']])
    
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
