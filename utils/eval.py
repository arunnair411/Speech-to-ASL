import numpy as np
import os, sys
from scipy.io import savemat, loadmat
import torch
import pdb

from .custom_losses import dice_coeff, psnr_loss


def eval_net(g_net, split_validation_filenames, params, dataset_deets, val_or_test_string, val_or_test_loader):
    # If test set results are to be stored, initialize the requisite variables
    if params['save_test_val_results'] is not False:
        chosen_float_precision = 'float32'
        data_dict = dict()
        data_dict['file_name'] = []                                     # Store the actual filename (including path) here; keep it empty for now, assign it later
        data_dict['file_name_idx'] = np.zeros((1, 1)).astype('uint32')  # Note: Not adding one or anything when storing... this is the actual number in the filename being read        
        data_dict['input_data'] = np.zeros((params['in_chans'],params['test_rows'],params['test_cols'])).astype(chosen_float_precision)
        data_dict['target_output'] = np.zeros((1,params['test_rows'],params['test_cols'])).astype('uint8')
        data_dict['DNN_output'] = np.zeros((1,params['test_rows'],params['test_cols'])).astype('uint8')

    tot_DSC = 0.
    tot_PSNR = 0.
    running_counter = 0
    with torch.no_grad():
    # with torch.set_grad_enabled(False): # Alternatively
        for i, sample in enumerate(val_or_test_loader):
            input_data = sample['input_data']
            target_output = sample['target_output']

            # Doing channel_data = channel_data.cuda() is the older way - can generalize it to arbitrary devices using .to()
            input_data = input_data.to(params['device'])
            target_output = target_output.to(params['device'])
            preds = g_net(input_data) 
            # test_start_time = timeit.default_timer()
            # for i in range(100):
                # preds_seg, preds_B = g_net(channel_data) #1*1*#n_rows*n_cols
            # elapsed = timeit.default_timer() - test_start_time
            # pdb.set_trace()
            
            preds = (torch.sigmoid(preds)>0.5).float() # TODO - sometimes thresholding helps... like 0.39, 0.391 instead of 0.5            

            tot_DSC += dice_coeff(preds, target_output).item() * input_data.shape[0] # mean * number of samples
            tot_PSNR += psnr_loss(preds, target_output).item() * input_data.shape[0] # mean * number of samples
            # tot += msssim(pred, beamformed_B, normalize=False) # TODO            
            running_counter = running_counter + input_data.shape[0]

            # If test set results are to be stored, update the requisite variables
            if params['save_test_val_results'] is not False:
                os.makedirs(params['results_dir'], exist_ok=True)
                output_dir = os.path.join(params['results_dir'], val_or_test_string)
                os.makedirs(output_dir, exist_ok=True)
                for test_idx in range(input_data.shape[0]):
                    # Assign the DNN inputs and outputs to the file
                    curr_input_data = input_data[test_idx].cpu().numpy()
                    data_dict['input_data'][0:params['in_chans']] = np.squeeze(curr_input_data).astype(chosen_float_precision)
                    curr_target_output = target_output[test_idx].cpu().numpy()
                    data_dict['target_output'][0:1] = np.squeeze(curr_target_output).astype(chosen_float_precision)

                    curr_pred = preds[test_idx].cpu().numpy()
                    data_dict['DNN_output'][0:1] = np.squeeze(curr_pred).astype('uint8')
                    # # Assign smaller identifying details to the file
                    data_dict['file_name'] = os.path.split(sample['file_name'][test_idx])[1]
                    data_dict['file_name_idx'] = sample['file_name_idx'][test_idx].numpy().astype('uint32') # Current iteration idx i.e. which file number is being processed. Filename it is stored as is this.
                    if 'batch_testing_id' in params.keys():
                        if not params['batch_testing_id']:
                            savemat(os.path.join(output_dir, '{}'.format(data_dict['file_name'])), data_dict)
                        elif params['batch_testing_id']: # Use batch_testing_id as a prefix to the file name
                            savemat(os.path.join(params['results_dir'], '{0:0>6d}_{1}'.format(params['batch_testing_id'], data_dict['file_name'])), data_dict)

    return (tot_DSC/running_counter , tot_PSNR/running_counter)
