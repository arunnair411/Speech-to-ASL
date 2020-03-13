import pdb
import torch
import torchvision

import pickle

def initialize_hooks_for_visualization(params, g_net):

    # Visualize feature maps 
    activation = {}
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook


    if params['normalization']=='none':
        hook_names = ['conv1_1', 'conv1_1_ReLU', 'conv1_2', 'conv1_2_ReLU',
                      'conv2_1', 'conv2_1_ReLU', 'conv2_2', 'conv2_2_ReLU',
                      'conv3_1', 'conv3_1_ReLU', 'conv3_2', 'conv3_2_ReLU',
                      'conv4_1', 'conv4_1_ReLU', 'conv4_2', 'conv4_2_ReLU',
                      'conv5_1', 'conv5_1_ReLU', 'conv5_2', 'conv5_2_ReLU',
                      'conv6_1', 'conv6_1_ReLU', 'conv6_2', 'conv6_2_ReLU',
                      'conv7_1', 'conv7_1_ReLU', 'conv7_2', 'conv7_2_ReLU',
                      'conv8_1', 'conv8_1_ReLU', 'conv8_2', 'conv8_2_ReLU',
                      'conv9_1', 'conv9_1_ReLU', 'conv9_2', 'conv9_2_ReLU',
                      'output_conv']
        num_block_elements = 4 # Each conv block posseses 4 elements, two conv layers and 2 ReLUs
    else:
        hook_names = ['conv1_1', 'conv1_1_N', 'conv1_1_ReLU', 'conv1_2', 'conv1_2_N', 'conv1_2_ReLU',
                      'conv2_1', 'conv2_1_N', 'conv2_1_ReLU', 'conv2_2', 'conv2_2_N', 'conv2_2_ReLU',
                      'conv3_1', 'conv3_1_N', 'conv3_1_ReLU', 'conv3_2', 'conv3_2_N', 'conv3_2_ReLU',
                      'conv4_1', 'conv4_1_N', 'conv4_1_ReLU', 'conv4_2', 'conv4_2_N', 'conv4_2_ReLU',
                      'conv5_1', 'conv5_1_N', 'conv5_1_ReLU', 'conv5_2', 'conv5_2_N', 'conv5_2_ReLU',
                      'conv6_1', 'conv6_1_N', 'conv6_1_ReLU', 'conv6_2', 'conv6_2_N', 'conv6_2_ReLU',
                      'conv7_1', 'conv7_1_N', 'conv7_1_ReLU', 'conv7_2', 'conv7_2_N', 'conv7_2_ReLU',
                      'conv8_1', 'conv8_1_N', 'conv8_1_ReLU', 'conv8_2', 'conv8_2_N', 'conv8_2_ReLU',
                      'conv9_1', 'conv9_1_N', 'conv9_1_ReLU', 'conv9_2', 'conv9_2_N', 'conv9_2_ReLU',
                      'output_conv']        
        num_block_elements = 6 # Each conv block posseses 6 elements, two conv layers, two normalization layers, and 2 ReLUs
    handles = []
    if params['fair']: 
        # if torch.nn.DataParallel isn't called, there is no ".module." in the name
        if params['no_parallel']:
            # Extract weights for the first filter - this visualizes the filter weights itself
            first_filter_weights = g_net.down_sample_layers[0].conv[0].weight.data
            first_filter_weights = first_filter_weights.reshape(first_filter_weights.shape[0]*first_filter_weights.shape[1],1,first_filter_weights.shape[2],first_filter_weights.shape[3])

            # Input (inc) Conv Block, down1 (down1) Conv Block, down2 (down2) Conv Block,
            # down3 (down3) Conv Block, down4 (down4) Conv Block, upx1 (upx1) Conv Block,
            # upx2 (upx2) Conv Block, upx3 (upx3) Conv Block, upx4 (upx4) Conv Block
            for idx in range(num_block_elements):
                for jdx in range(len(g_net.down_sample_layers)):
                    handles.append(g_net.down_sample_layers[jdx].conv[idx].register_forward_hook(get_activation(hook_names[idx+jdx*num_block_elements])))
                curr_base_count = len(g_net.down_sample_layers)*num_block_elements
                handles.append(g_net.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+curr_base_count])))
                curr_base_count += num_block_elements
                for jdx in range(len(g_net.up_sample_layers)):
                    handles.append(g_net.up_sample_layers[jdx].conv[idx].register_forward_hook(get_activation(hook_names[idx+jdx*num_block_elements+curr_base_count])))
            # Output (out) conv layer
            handles.append(g_net.conv2[0].register_forward_hook(get_activation(hook_names[-1])))
        # ... otherwise there is a ".module" in the name
        else:
            # Extract weights for the first filter - this visualizes the filter weights itself
            first_filter_weights = g_net.module.down_sample_layers[0].conv[0].weight.data
            first_filter_weights = first_filter_weights.reshape(first_filter_weights.shape[0]*first_filter_weights.shape[1],1,first_filter_weights.shape[2],first_filter_weights.shape[3])

            # Input (inc) Conv Block, down1 (down1) Conv Block, down2 (down2) Conv Block,
            # down3 (down3) Conv Block, down4 (down4) Conv Block, upx1 (upx1) Conv Block,
            # upx2 (upx2) Conv Block, upx3 (upx3) Conv Block, upx4 (upx4) Conv Block
            for idx in range(num_block_elements):
                for jdx in range(len(g_net.module.down_sample_layers)):
                    handles.append(g_net.module.down_sample_layers[jdx].conv[idx].register_forward_hook(get_activation(hook_names[idx+jdx*num_block_elements])))
                curr_base_count = len(g_net.module.down_sample_layers)*num_block_elements
                handles.append(g_net.module.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+curr_base_count])))
                curr_base_count += num_block_elements
                for jdx in range(len(g_net.module.up_sample_layers)):
                    handles.append(g_net.module.up_sample_layers[jdx].conv[idx].register_forward_hook(get_activation(hook_names[idx+jdx*num_block_elements+curr_base_count])))
            # Output (out) conv layer
            handles.append(g_net.module.conv2[0].register_forward_hook(get_activation(hook_names[-1])))

    else:
        # if torch.nn.DataParallel isn't called, there is no ".module." in the name
        if params['no_parallel']:
            # Extract weights for the first filter - this visualizes the filter weights itself
            first_filter_weights = g_net.inc.conv.conv[0].weight.data
            first_filter_weights = first_filter_weights.reshape(first_filter_weights.shape[0]*first_filter_weights.shape[1],1,first_filter_weights.shape[2],first_filter_weights.shape[3])

            # Input (inc) Conv Block, down1 (down1) Conv Block, down2 (down2) Conv Block,
            # down3 (down3) Conv Block, down4 (down4) Conv Block, upx1 (upx1) Conv Block,
            # upx2 (upx2) Conv Block, upx3 (upx3) Conv Block, upx4 (upx4) Conv Block
            for idx in range(num_block_elements):
                handles.append(g_net.inc.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx])))
                handles.append(g_net.down1.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+1*num_block_elements])))
                handles.append(g_net.down2.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+2*num_block_elements])))
                handles.append(g_net.down3.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+3*num_block_elements])))
                handles.append(g_net.down4.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+4*num_block_elements])))
                handles.append(g_net.upx1.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+5*num_block_elements])))
                handles.append(g_net.upx2.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+6*num_block_elements])))
                handles.append(g_net.upx3.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+7*num_block_elements])))
                handles.append(g_net.upx4.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+8*num_block_elements])))
            # Output (out) conv layer
            handles.append(g_net.outcx.conv.register_forward_hook(get_activation(hook_names[-1])))

        # ... otherwise there is a ".module" in the name
        else:
            # Extract weights for the first filter - this visualizes the filter weights itself
            first_filter_weights = g_net.module.inc.conv.conv[0].weight.data
            first_filter_weights = first_filter_weights.reshape(first_filter_weights.shape[0]*first_filter_weights.shape[1],1,first_filter_weights.shape[2],first_filter_weights.shape[3])

            # Input (inc) Conv Block, down1 (down1) Conv Block, down2 (down2) Conv Block,
            # down3 (down3) Conv Block, down4 (down4) Conv Block, upx1 (upx1) Conv Block,
            # upx2 (upx2) Conv Block, upx3 (upx3) Conv Block, upx4 (upx4) Conv Block
            for idx in range(num_block_elements):
                handles.append(g_net.module.inc.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx])))
                handles.append(g_net.module.down1.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+1*num_block_elements])))
                handles.append(g_net.module.down2.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+2*num_block_elements])))
                handles.append(g_net.module.down3.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+3*num_block_elements])))
                handles.append(g_net.module.down4.mpconv[1].conv[idx].register_forward_hook(get_activation(hook_names[idx+4*num_block_elements])))
                handles.append(g_net.module.upx1.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+5*num_block_elements])))
                handles.append(g_net.module.upx2.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+6*num_block_elements])))
                handles.append(g_net.module.upx3.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+7*num_block_elements])))
                handles.append(g_net.module.upx4.conv.conv[idx].register_forward_hook(get_activation(hook_names[idx+8*num_block_elements])))
            # Output (out) conv layer
            handles.append(g_net.module.outcx.conv.register_forward_hook(get_activation(hook_names[-1])))

    return g_net, activation, hook_names, num_block_elements, handles, first_filter_weights

# -------------------------------------------------------------------------------------------------
## Visualization function for neuron activations -- track it using tensorboard
# From http://cs231n.github.io/understanding-cnn/ - Nice concise exposition of work in the field
# One dangerous pitfall that can be easily noticed with this visualization is that some activation 
# maps may be all zero **for many different inputs**, which can indicate dead filters, and can be a 
# symptom of high learning rates.
# Activations ARE often naturally sparse...
# https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e        
def visualize_neurons(params, epoch, g_net, data_loader, writer):
    def save_image(image, tag, nrow):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=nrow, pad_value=1)
        writer.add_image(tag, grid, epoch)

    g_net.eval()
    # One way to copy a model - https://discuss.pytorch.org/t/check-if-model-is-eval-or-train/9395 - there might be other better ways
    # TODO: This is done to solve the issue of subsequent training not doing as well when I visualize the model weights...
    copied_model = pickle.loads(pickle.dumps(g_net))    
    with torch.no_grad():
        # for iterIdx, sample in enumerate(data_loader): # TODO: if using this, uncomment break at the bottom, and tabspace everything right
        sample = next(iter(data_loader))
        # Register hooks to visualize the feature maps
        copied_model, activation, hook_names, num_block_elements, handles, first_filter_weights = initialize_hooks_for_visualization(params, copied_model)

        input_data = sample['input_data'].to(params['device']) 
        target_output = sample['target_output'].to(params['device']) 
        
        preds = copied_model(input_data)
        preds = torch.sigmoid(preds)

        overall_count = 0
        block_count = 0
        while overall_count < len(hook_names)-1:
            for i in range(num_block_elements):
                act = activation[hook_names[overall_count+i]].squeeze()
                act = act[0,:,:,:].unsqueeze(1) # Make each filter activation for the first input image behave like one image of a batch
                save_image(act, f"Features/ConvBlock_{block_count}/{hook_names[overall_count+i]}", nrow=8)
            overall_count +=num_block_elements
            block_count+=1
        # Finally, record the output block's output
        act = activation[hook_names[-1]].squeeze()
        act = act[0,:,:].unsqueeze(0).unsqueeze(0) # Different unsqueezing now. NOTE: Remember the headscratching 16/32 because of multi-GPU
        save_image(act, f"Features/OutputConvBlock/{hook_names[-1]}", nrow=8)

        # Save a visualization of the weights learned at layer 1
        save_image(first_filter_weights, f"Features/FirstLayerWeights", nrow=8)

        # Save an image of the output, target, and error too Just in case
        chosen_target = target_output[0,:,:,:].unsqueeze(1)
        chosen_pred = preds[0,:,:,:].unsqueeze(1)
        chosen_error = torch.abs(target_output - preds)[0,:,:,:].unsqueeze(1)
        chosen_triplet = torch.cat((chosen_target, chosen_pred, chosen_error), dim=0)
        save_image(chosen_triplet, 'Features/Images/target_pred_error', nrow=3)

        # Need to remove the created handles or else it'll blow up the memory
        for handle in handles:
            handle.remove()            
        
        del copied_model
        
        # break        