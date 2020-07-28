import os

# # -------------------------------------------------------------------------------------------------
# ## Utility functions
# def retrieve_dataset_filenames(dataset='DNS-challenge-synthetic-test'):
#     dataset_deets = dict()
#     dataset_prefix = '/home/t-arnair/projects/DNS-Challenge/'
#     if dataset == 'DNS-challenge-synthetic-test':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/OG/clean')]
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/OG/noisy')]
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/noisy')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]   # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/noisy')]   # Setting with_reverb part of the synthetic test set as test_data
#     elif dataset == 'DNS-challenge-synthetic-test-expanded':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean')]
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/noisy')]
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/noisy')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]   # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/noisy')]   # Setting with_reverb part of the synthetic test set as test_data
#     elif dataset == 'DNS-challenge-synthetic-test-expanded-cleanvoip':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean')]
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/noisy')]
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/noisy')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]   # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/noisy')]   # Setting with_reverb part of the synthetic test set as test_data
#     elif dataset == 'DNS-challenge-synthetic-test-expanded-cleanteledonline':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean')]
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/noisy')]
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/noisy')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]   # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/noisy')]   # Setting with_reverb part of the synthetic test set as test_data        
#     elif dataset == 'kaz':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'datasets/clean')]
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'datasets/noise')] # NOTE: Technically, this is the noise and not the noisy data since noisy data is mixed on the fly...
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/noisy')]     # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]   # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']  = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/noisy')]   # Setting with_reverb part of the synthetic test set as test_data
#     elif dataset == 'DNS-challenge-synthetic-test-expanded-cleanteledstored':
#         dataset_deets['dataset'] = dataset
#         dataset_deets['clean_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean'),
#                                                         os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean'),
#                                                         os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean')] # Repeat it thrice since we have three sets of noisy data for each clean file
#         dataset_deets['noisy_train_data_path_list'] = [os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean_amr'),
#                                                         os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean_mp3'),
#                                                         os.path.join(dataset_prefix, 'Arun_train_datasets/Expanded/clean_mulaw')]
#         dataset_deets['clean_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean')]         # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['noisy_val_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean_amr'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean_mp3'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/no_reverb/clean_mulaw')]   # Setting no_reverb part of the synthetic test set as val_data
#         dataset_deets['clean_test_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean')]       # Setting with_reverb part of the synthetic test set as test_data
#         dataset_deets['noisy_test_data_path_list']   = [os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean_amr'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean_mp3'),
#                                                         os.path.join(dataset_prefix,'datasets/test_set/synthetic/with_reverb/clean_mulaw')] # Setting with_reverb part of the synthetic test set as test_data
#     else:
#         dataset_deets['dataset'] = 'currRealData'.lower() # aka currrealdata
#         dataset_deets['clean_train_data_path_list'] = []
#         dataset_deets['noisy_train_data_path_list'] = []
#         dataset_deets['clean_val_data_path_list'] = []
#         dataset_deets['noisy_val_data_path_list'] = []
#         dataset_deets['clean_test_data_path_list'] = [] 
#         dataset_deets['noisy_test_data_path_list'] = [dataset] # Directly point to directory containing the data (As a list)
#     return dataset_deets

def retrieve_dataset_filenames(dataset='pose_npy_dir'):
    dataset_deets = dict()
    dataset_prefix = '/data2/t-arnair/projects/Speech-to-ASL/Gloss2Avatar/data/RawData'
    if dataset == 'pose_npy_dir':
        dataset_deets['dataset'] = dataset
        dataset_deets['data_path_list'] = [os.path.join(dataset_prefix, 'pose_npy_dir')]
    else:
        dataset_deets['dataset'] = 'currRealData'.lower() # aka currrealdata
        dataset_deets['data_path_list'] = [dataset]
    return dataset_deets
