import numpy as np
import torch
import torch.nn
import os, sys
from scipy.io import wavfile
import torch
import pdb
import soundfile as sf
import librosa
import tempfile
from subprocess import run, PIPE
import re
from itertools import repeat

import multiprocessing
from multiprocessing import Pool
PROCESSES = multiprocessing.cpu_count()//2

from .custom_losses import dice_coeff, psnr_loss, RMSELoss, LSDLoss

def eval_net_pose_predictor(g_net, criterion_g, split_filenames, params, dataset_deets, val_or_test_string, val_or_test_loader):
    # If test set results are to be stored, initialize the requisite variables
    
    tot_MAELoss = 0.
    criterion_mae = torch.nn.L1Loss()
    tot_MSELoss = 0.
    criterion_mse = torch.nn.MSELoss()
    tot_RMSELoss = 0.
    criterion_rmse = RMSELoss()
    tot_loss = 0.
    running_counter = 0
    with torch.no_grad():
    # with torch.set_grad_enabled(False): # Alternatively
        for i, sample in enumerate(val_or_test_loader):
            print(f"Current Test Idx is {i}")
            data = sample['data']

            edge_sample_length = 1 # MATCH TO train_v5.py, test_v5.py
            sample_gap = 10 # MATCH TO train_v5.py, test_v5.py            
            num_input_frames = 2*edge_sample_length
            num_output_frames = sample_gap
            total_seq_frames = target_output.shape[1]
            input_data = torch.zeros(total_seq_frames-num_output_frames-num_input_frames+1, num_input_frames, 137, 3)
            target_output = torch.zeros(total_seq_frames-num_output_frames-num_input_frames+1, num_output_frames, 137, 3)
            if params['architecture'] in ['PosePredictorFC']:
                for loop_idx in range(total_seq_frames-num_output_frames-num_input_frames+1): # Need to process the last frame separately
                    relevant_data = data[:,loop_idx:loop_idx+num_output_frames+num_input_frames,:,:]
                    target_output_single = relevant_data[:, edge_sample_length:-edge_sample_length, :, : ] # TODO: Check if you need detach() and clone() .detach().clone()
                    input_data_single = torch.cat((relevant_data[:, :edge_sample_length, :, :], relevant_data[:, -edge_sample_length:, :, :]), dim=1)
                    input_data[loop_idx, :, :, :] = input_data_single
                    target_output[loop_idx, :, :, :] = target_output_single
                
                # Doing channel_data = channel_data.cuda() is the older way - can generalize it to arbitrary devices using .to()
                input_data = input_data.to(params['device'])
                target_output = target_output.to(params['device'])
                preds = g_net(input_data)
                

            # Obtain the clean and degraded waveforms            
            # tot_PSNR += psnr_loss(preds, target_output).item() * input_data.shape[0] # mean * number of samples # Occasionally throwing an erorr, got annoyed, setting it to 0
            tot_loss += criterion_g(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
            tot_MAELoss += criterion_mae(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
            tot_MSELoss += criterion_mse(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
            tot_RMSELoss+= criterion_rmse(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
            # # tot += msssim(pred, beamformed_B, normalize=False) 
            running_counter = running_counter + input_data.shape[0]

            # TODO: Have to implement the writing it to
            # multi_pool = multiprocessing.Pool(processes=PROCESSES)
            # input_data_cpu    = input_data.cpu().numpy()
            # preds_cpu         = preds.cpu().numpy()
            # target_output_cpu = target_output.cpu().numpy()
            # file_name_transfer = sample['file_name']
            # params
            # if 'phase_noisy_fft_data' in sample.keys():
            #     phase_noisy_fft_data = sample['phase_noisy_fft_data'].cpu().numpy()
            #     # SPECIAL FUNCTION IF ESTIMATING THE MISSING FRAME MASK...
            #     # out_list = np.array(multi_pool.starmap(inner_loop_fn_voip_2, zip(input_data_cpu, preds_cpu, target_output_cpu, file_name_transfer, repeat(params), repeat(val_or_test_string), phase_noisy_fft_data)))
            #     out_list = np.array(multi_pool.starmap(inner_loop_fn, zip(input_data_cpu, preds_cpu, target_output_cpu, file_name_transfer, repeat(params), repeat(val_or_test_string), phase_noisy_fft_data)))
            # else:
            #     out_list = np.array(multi_pool.starmap(inner_loop_fn, zip(input_data_cpu, preds_cpu, target_output_cpu, file_name_transfer, repeat(params), repeat(val_or_test_string),)))
            # # Close the parallel pool
            # multi_pool.close()
            # multi_pool.join()
            if params['save_test_val_results'] is not False:
                # 0 - Create results_dir to output files to                
                output_dir = os.path.join(params['results_dir'], val_or_test_string)
                os.makedirs(output_dir, exist_ok=True)                                
                # 1 - Write the groundtruth file
                output_filename_1 = os.path.join(output_dir, "".join(('groundtruth_', os.path.basename(file_name))))
                np.save(output_filename_1, target_output.cpu().numpy())
                # 2 - Write the predicted file
                output_filename_2 = os.path.join(output_dir, "".join(('predicted_', os.path.basename(file_name))))
                np.save(output_filename_2, preds.cpu().numpy())

    return (tot_MAELoss/running_counter, tot_MSELoss/running_counter, tot_RMSELoss/running_counter, tot_loss/running_counter)

def inner_loop_fn(input_data, preds, target_output, file_name, params, val_or_test_string, phase_noisy_fft_data=None):
    PESQ_DIR = '/home/t-arnair/projects/Dung/tf_se/se/evaluation/ITU_PESQ/Software/P862_annex_A_2005_CD/source'
    # 0 - Generate reference, noisy, and denoised time domain signals
    if params['architecture'] in params['time_domain_architectures']: # Time Domain networks
        ref_wav = np.squeeze(target_output)
        noisy_wav = input_data[0,:]
        denoised_wav  = np.squeeze(preds)
    else:
        noisy_stft    = np.multiply(np.pad( np.sqrt(np.exp(input_data[0,:,:])), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(phase_noisy_fft_data)))
        noisy_wav     = librosa.istft(noisy_stft, win_length = 512, hop_length = 256, window='hann')
        denoised_stft = np.multiply(np.pad( np.sqrt(np.exp(np.squeeze(preds))), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(phase_noisy_fft_data)))
        denoised_wav  = librosa.istft(denoised_stft, win_length = 512, hop_length = 256, window='hann')
        clean_filename = os.path.join('/'.join(os.path.dirname(file_name).split('/')[:-1]), 'clean','clean_fileid_'+os.path.basename(file_name).split('fileid_')[1])
        if os.path.isfile(clean_filename):
            ref_wav, ref_sr = sf.read(clean_filename)
            if ref_sr != 16000:
                ref_wav = librosa.resample(ref_wav, ref_sr, 16000)
        else:
            ref_wav = np.zeros_like(noisy_wav)                    
    # 1 - Calculate PESQ if we do nothing
    pesq_raw = PESQ(ref_wav, noisy_wav, PESQ_DIR)
    if 'error!' not in pesq_raw:
        PESQ_donothing = float(PESQ(ref_wav, noisy_wav, PESQ_DIR))
    else:
        pass # Do nothing
    # 2 - Calculate PESQ for network processed output                    
    pesq_raw = PESQ(ref_wav, denoised_wav, PESQ_DIR)
    if 'error!' not in pesq_raw:
        PESQ_enhanced = float(PESQ(ref_wav, denoised_wav, PESQ_DIR))
    else:
        pass # Do nothing
    # 3 - If test set results are to be stored, update the requisite variables
    if params['save_test_val_results'] is not False:
        # 0 - Create results_dir to output files to
        os.makedirs(params['results_dir'], exist_ok=True)
        os.makedirs(os.path.join(params['results_dir'], val_or_test_string), exist_ok=True)
        output_dir = os.path.join(params['results_dir'], val_or_test_string, os.path.dirname(file_name).split(os.sep)[-1])
        os.makedirs(output_dir, exist_ok=True)
        # 1 - Write the denoised file
        datacont = (denoised_wav*32767)
        datacont = datacont.astype('int16')
        output_filename_1 = os.path.join(output_dir, "".join(('denoised_', os.path.basename(file_name))))
        wavfile.write(output_filename_1, 16000, datacont)
        # 2 - Write the noisy file
        datacont = (noisy_wav*32767)
        datacont = datacont.astype('int16')
        output_filename_2 = os.path.join(output_dir, "".join(('noisy_', os.path.basename(file_name))))
        wavfile.write(output_filename_2, 16000, datacont)    
    return (PESQ_donothing, PESQ_enhanced)

def inner_loop_fn_voip_2(input_data, preds, target_output, file_name, params, val_or_test_string, phase_noisy_fft_data=None):
    PESQ_DIR = '/home/t-arnair/projects/Dung/tf_se/se/evaluation/ITU_PESQ/Software/P862_annex_A_2005_CD/source'
    # 0 - Generate reference, noisy, and denoised time domain signals
    noisy_stft    = np.multiply(np.pad( np.sqrt(np.exp(np.squeeze(input_data))), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(phase_noisy_fft_data)))
    noisy_wav     = librosa.istft(noisy_stft, win_length = 512, hop_length = 256, window='hann')
    
    # Use noisy_wav to look at silences and missing packets, and use that information to only copy paste network output where required
    intervals_silence = librosa.effects.split(y=noisy_wav, frame_length = 512, hop_length=256, top_db = 30)
    Y2_est = np.zeros_like(noisy_wav) # NOTE: One difference - this is estimated from noisy signal, while the original is estimated from the clean signal... see if it hurts
    for curr_interval in intervals_silence:
        Y2_est[curr_interval[0]:curr_interval[1]] = 1
    
    _sr = 16000
    voip_packet_time = 0.010 # in [s] # This allows us to simulate packet sizes of different durations
    voip_packet_samples = int(_sr*voip_packet_time)
    intervals_missedpackets = librosa.effects.split(y=noisy_wav, frame_length = voip_packet_samples//2, hop_length=voip_packet_samples//4, top_db = 60)
    Y6 = np.zeros_like(noisy_wav)
    for curr_interval in intervals_missedpackets:
        Y6[curr_interval[0]:curr_interval[1]] = 1
    Y8 = 1- (1-Y6)*Y2_est # This is a mask, ignoring silence regions, which is 0 only in packet loss regions and 1 everywhere else (silence+normal signal)

    # Take STFT of Y8 to generate a mask for the STFT
    Y9 = np.abs(librosa.stft(10000*(1-Y8), n_fft=512, hop_length=256, window='hann'))
    Y9_summed = np.sum(Y9, axis=0) # This tells us all the frames affected by packet loss
    Y9_summed[Y9_summed>0]=1
    Y9_mask = np.tile(Y9_summed, (256,1))    

    preds_stitched = Y9_mask*np.squeeze(preds) + (1-Y9_mask)*np.squeeze(input_data)
    preds = preds_stitched

    denoised_stft = np.multiply(np.pad( np.sqrt(np.exp(np.squeeze(preds))), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(phase_noisy_fft_data)))
    denoised_wav  = librosa.istft(denoised_stft, win_length = 512, hop_length = 256, window='hann')
    clean_filename = os.path.join('/'.join(os.path.dirname(file_name).split('/')[:-1]), 'clean','clean_fileid_'+os.path.basename(file_name).split('fileid_')[1])
    if os.path.isfile(clean_filename):
        ref_wav, ref_sr = sf.read(clean_filename)
        if ref_sr != 16000:
            ref_wav = librosa.resample(ref_wav, ref_sr, 16000)
    else:
        ref_wav = np.zeros_like(noisy_wav)                    
    # 1 - Calculate PESQ if we do nothing
    pesq_raw = PESQ(ref_wav, noisy_wav, PESQ_DIR)
    if 'error!' not in pesq_raw:
        PESQ_donothing = float(PESQ(ref_wav, noisy_wav, PESQ_DIR))
    else:
        pass # Do nothing
    # 2 - Calculate PESQ for network processed output                    
    pesq_raw = PESQ(ref_wav, denoised_wav, PESQ_DIR)
    if 'error!' not in pesq_raw:
        PESQ_enhanced = float(PESQ(ref_wav, denoised_wav, PESQ_DIR))
    else:
        pass # Do nothing
    # 3 - If test set results are to be stored, update the requisite variables
    if params['save_test_val_results'] is not False:
        # 0 - Create results_dir to output files to
        os.makedirs(params['results_dir'], exist_ok=True)
        os.makedirs(os.path.join(params['results_dir'], val_or_test_string), exist_ok=True)
        output_dir = os.path.join(params['results_dir'], val_or_test_string, os.path.dirname(file_name).split(os.sep)[-1])
        os.makedirs(output_dir, exist_ok=True)
        # 1 - Write the denoised file
        datacont = (denoised_wav*32767)
        datacont = datacont.astype('int16')
        output_filename_1 = os.path.join(output_dir, "".join(('denoised_', os.path.basename(file_name))))
        wavfile.write(output_filename_1, 16000, datacont)
        # 2 - Write the noisy file
        datacont = (noisy_wav*32767)
        datacont = datacont.astype('int16')
        output_filename_2 = os.path.join(output_dir, "".join(('noisy_', os.path.basename(file_name))))
        wavfile.write(output_filename_2, 16000, datacont)    
    return (PESQ_donothing, PESQ_enhanced)   

# def eval_net_old(g_net, criterion_g, split_validation_filenames_clean, split_validation_filenames_noisy, params, dataset_deets, val_or_test_string, val_or_test_loader):
#     # If test set results are to be stored, initialize the requisite variables
    
#     PESQ_DIR = '/home/t-arnair/projects/Dung/tf_se/se/evaluation/ITU_PESQ/Software/P862_annex_A_2005_CD/source'
#     tot_PESQ_donothing = 0.
#     tot_PESQ = 0.
#     tot_PSNR = 0.
#     tot_MAELoss = 0.
#     criterion_mae = torch.nn.L1Loss()
#     tot_MSELoss = 0.
#     criterion_mse = torch.nn.MSELoss()
#     tot_RMSELoss = 0.
#     criterion_rmse = RMSELoss()
#     tot_loss = 0.
#     running_counter = 0
#     with torch.no_grad():
#     # with torch.set_grad_enabled(False): # Alternatively
#         for i, sample in enumerate(val_or_test_loader):
#             print(f"Current Test Idx is {i}")
#             input_data = sample['input_data']
#             target_output = sample['target_output']

#             # Doing channel_data = channel_data.cuda() is the older way - can generalize it to arbitrary devices using .to()
#             input_data = input_data.to(params['device'])
#             target_output = target_output.to(params['device'])
#             num_frames = target_output.shape[-1]
#             if params['architecture'] in ['unet1dk15s4', 'unet1dk5s2']: # Time Domain networks
#                 preds         = torch.zeros_like(target_output) # Don't need to do .to(device), already there
#                 frame_length = 16384
#                 for loop_idx in range((num_frames-1)//frame_length): # Need to process the last frame separately
#                     preds[:,:,loop_idx*frame_length:(loop_idx+1)*frame_length] = g_net(input_data[:,:,loop_idx*frame_length:(loop_idx+1)*frame_length])                
#                 preds[:,:,num_frames-frame_length:num_frames] = g_net(input_data[:,:,num_frames-frame_length:num_frames])
#             elif params['data_loader_size'] in ['onesecond', 'quartersecond']:
#                 preds         = torch.zeros_like(target_output) # Don't need to do .to(device), already there
#                 frame_length = 64 if params['data_loader_size']=='onesecond' else 16
#                 for loop_idx in range((num_frames-1)//frame_length): # Need to process the last frame separately
#                     preds[:,:,:,loop_idx*frame_length:(loop_idx+1)*frame_length] = g_net(input_data[:,:,:,loop_idx*frame_length:(loop_idx+1)*frame_length]) # NOTE: Assuming test samples have same number of STFT frames - during training, it's fine and during testing, set test-batch-size=1 to not bother
#                 preds[:,:,:,num_frames-frame_length:num_frames] = g_net(input_data[:,:,:,num_frames-frame_length:num_frames])
            
#             # test_start_time = timeit.default_timer()
#             # for i in range(100):
#                 # preds_seg, preds_B = g_net(channel_data) #1*1*#n_rows*n_cols
#             # elapsed = timeit.default_timer() - test_start_time

#             # Obtain the clean and degraded waveforms            
#             tot_PSNR += psnr_loss(preds, target_output).item() * input_data.shape[0] # mean * number of samples
#             tot_loss += criterion_g(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
#             tot_MAELoss += criterion_mae(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
#             tot_MSELoss += criterion_mse(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
#             tot_RMSELoss+= criterion_rmse(preds.view(-1), target_output.view(-1)).item() * input_data.shape[0] # mean * number of samples
#             for test_idx in range(input_data.shape[0]):
#                 # 0 - Generate reference, noisy, and denoised time domain signals
#                 if params['architecture'] in ['unet1dk15s4', 'unet1dk5s2']: # Time Domain networks                    
#                     ref_wav = np.squeeze(target_output[test_idx].cpu().numpy())
#                     noisy_wav = np.squeeze(input_data[test_idx].cpu().numpy())
#                     denoised_wav  = np.squeeze(preds[test_idx].cpu().numpy())
#                 else:
#                     noisy_stft    = np.multiply(np.pad( np.sqrt(np.exp(np.squeeze(input_data[test_idx].cpu().numpy()))), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(sample['phase_noisy_fft_data'][test_idx].cpu().numpy())))
#                     noisy_wav     = librosa.istft(noisy_stft, win_length = 512, hop_length = 256, window='hann')
#                     denoised_stft = np.multiply(np.pad( np.sqrt(np.exp(np.squeeze(preds[test_idx].cpu().numpy()))), ((0,1),(0,0)), 'constant'), np.exp(1j * np.squeeze(sample['phase_noisy_fft_data'][test_idx].cpu().numpy())))
#                     denoised_wav  = librosa.istft(denoised_stft, win_length = 512, hop_length = 256, window='hann')
#                     clean_filename = os.path.join('/'.join(os.path.dirname(sample['file_name'][test_idx]).split('/')[:-1]), 'clean','clean_fileid_'+os.path.basename(sample['file_name'][test_idx]).split('fileid_')[1])
#                     if os.path.isfile(clean_filename):
#                         ref_wav, ref_sr = sf.read(clean_filename)
#                         if ref_sr != 16000:
#                             ref_wav = librosa.resample(ref_wav, ref_sr, 16000)
#                     else:
#                         ref_wav = np.zeros_like(noisy_wav)                    
#                 # 1 - Calculate PESQ if we do nothing
#                 pesq_raw = PESQ(ref_wav, noisy_wav, PESQ_DIR)
#                 if 'error!' not in pesq_raw:
#                     tot_PESQ_donothing += float(PESQ(ref_wav, noisy_wav, PESQ_DIR))
#                 else:
#                     pass # Do nothing
#                 # 2 - Calculate PESQ for network processed output                    
#                 pesq_raw = PESQ(ref_wav, denoised_wav, PESQ_DIR)
#                 if 'error!' not in pesq_raw:
#                     tot_PESQ += float(PESQ(ref_wav, denoised_wav, PESQ_DIR))
#                 else:
#                     pass # Do nothing
#                 # 3 - If test set results are to be stored, update the requisite variables
#                 if params['save_test_val_results'] is not False:
#                     # 0 - Create results_dir to output files to
#                     os.makedirs(params['results_dir'], exist_ok=True)
#                     output_dir = os.path.join(params['results_dir'], val_or_test_string)
#                     os.makedirs(output_dir, exist_ok=True)
#                     # 1 - Write the denoised file
#                     datacont = (denoised_wav*32767)
#                     datacont = datacont.astype('int16')
#                     output_filename_1 = os.path.join(output_dir, "".join(('denoised_', os.path.basename(sample['file_name'][test_idx])))     )
#                     wavfile.write(output_filename_1, 16000, datacont)
#                     # 2 - Write the noisy file
#                     datacont = (noisy_wav*32767)
#                     datacont = datacont.astype('int16')
#                     output_filename_2 = os.path.join(output_dir, "".join(('noisy_', os.path.basename(sample['file_name'][test_idx])))     )
#                     wavfile.write(output_filename_2, 16000, datacont)
#                 # tot += msssim(pred, beamformed_B, normalize=False) # TODO
#             running_counter = running_counter + input_data.shape[0]
#     return (tot_PESQ_donothing/running_counter, tot_PESQ/running_counter , tot_PSNR/running_counter, tot_MAELoss/running_counter, tot_MSELoss/running_counter, tot_RMSELoss/running_counter, tot_loss/running_counter)
