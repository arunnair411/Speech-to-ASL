import os

# -------------------------------------------------------------------------------------------------
## Utility functions
def retrieve_dataset_filenames(dataset='20200102_Sim1'):
    dataset_deets = dict()
    dataset_prefix = '../Simulator'
    if dataset == '20200102_Sim1_0.20':
        dataset_deets['dataset'] = dataset
        dataset_deets['train_data_path_list'] = [os.path.join(dataset_prefix,'20200102_Sim1_0.20/train/')]
        dataset_deets['val_data_path_list'] =   [os.path.join(dataset_prefix,'20200102_Sim1_0.20/val/')]
        dataset_deets['test_data_path_list']  = [os.path.join(dataset_prefix,'20200102_Sim1_0.20/test/')]
    elif dataset == '20200102_Sim2_0.10':
        dataset_deets['dataset'] = dataset
        dataset_deets['train_data_path_list'] = [os.path.join(dataset_prefix,'20200102_Sim2_0.10/train/')]
        dataset_deets['val_data_path_list'] =   [os.path.join(dataset_prefix,'20200102_Sim2_0.10/val/')]
        dataset_deets['test_data_path_list']  = [os.path.join(dataset_prefix,'20200102_Sim2_0.10/test/')]        
    elif dataset == '20200102_Sim3_0.05':
        dataset_deets['dataset'] = dataset
        dataset_deets['train_data_path_list'] = [os.path.join(dataset_prefix,'20200102_Sim3_0.05/train/')]
        dataset_deets['val_data_path_list'] =   [os.path.join(dataset_prefix,'20200102_Sim3_0.05/val/')]
        dataset_deets['test_data_path_list']  = [os.path.join(dataset_prefix,'20200102_Sim3_0.05/test/')]
    else:
        dataset_deets['dataset'] = 'currRealData'.lower() # aka currrealdata
        dataset_deets['train_data_path_list'] = []
        dataset_deets['val_data_path_list'] = []
        dataset_deets['test_data_path_list'] = [dataset] # Directly point to directory containing the data (As a list)
    return dataset_deets
