import numpy as np
from scipy import signal
import scipy.ndimage

def get_deconve_mask(W, layer_names, SRC, depth):
    def relu(x):
        return np.maximum(0., x)

    def get_deconvolve(images, weights):
        num_before_deconv, num_after_deconv, num_row, num_col = weights.shape
        flipped_weights = weights[:, :, ::-1, ::-1]
        reversed_flipped_weights = np.zeros((num_after_deconv, num_before_deconv, num_row, num_col))
        for dim0 in range(num_after_deconv):
            for dim1 in range(num_before_deconv):
                reversed_flipped_weights[dim0, dim1, :, :] = flipped_weights[dim1, dim0, :, :]
        
        return get_convolve(images, reversed_flipped_weights)

    def get_unpooling2d(images, switches, ds=2):
        num_image, num_img_row, num_img_col = images.shape
        num_switch, num_swt_row, num_swt_col = switches.shape
        out_images = np.zeros((num_image, num_swt_row, num_swt_col))
        for ind_image, image in enumerate(images):
            out_images[ind_image, :num_img_row*ds, :num_img_col*ds] = np.multiply(scipy.ndimage.zoom(image, ds, order=0), switches[ind_image, :num_img_row*ds, :num_img_col*ds])
        return out_images

    def get_convolve(images, weights):
        num_out, num_in, num_row_w, num_col_w = weights.shape
        num_row_img, num_col_img = images.shape[1], images.shape[2]
        out_images = np.zeros((num_out, num_row_img, num_col_img))
        
        for ind_input_layer in range(weights.shape[1]):
            for ind_output_layer in range(weights.shape[0]):
                out_images[ind_output_layer, :, :] += signal.convolve2d(images[ind_input_layer, :, :], weights[ind_output_layer, ind_input_layer, :, :], mode='same')

        return out_images

    def get_MP2d(images, ds=2):
        num_image, num_row, num_col = images.shape
        out_images = np.zeros((num_image, num_row//ds, num_col//ds))
        switch = np.zeros((num_image, num_row, num_col))
        
        for ind_image, image in enumerate(images):
            for row_ind in range(num_row//ds):
                for col_ind in range(num_col//ds):
                    out_images[ind_image, row_ind, col_ind] = np.max(image[ds*row_ind:ds*row_ind+ds, ds*col_ind:ds*col_ind+ds])
                    argmax_here = np.argmax(image[ds*row_ind:ds*row_ind+ds, ds*col_ind:ds*col_ind+ds])
                    switch[ind_image, ds*row_ind+argmax_here//ds, ds*col_ind+argmax_here%ds] = 1

        return out_images, switch

    MAG = []
    MAG.append(np.zeros((1, SRC.shape[0], SRC.shape[1])))
    
    MAG[0][0,:,:] = np.abs(SRC)
    
    switch_matrices = []
    procedures = []
    conv_ind = 0
    mp_ind = 0
    
    print('-------feed-forward-')
    for layer_ind, layer_name in enumerate(layer_names):
        if layer_name == "Convolution2D":
            MAG.append(relu(get_convolve(images=MAG[-1], weights=W[conv_ind])))
            procedures.append('conv')
            conv_ind += 1

        elif layer_name == "MaxPooling2D":
            result, switch = get_MP2d(images=MAG[-1], ds=2)
            MAG.append(result)
            procedures.append('MP')
            switch_matrices.append(switch)
            mp_ind += 1

        if mp_ind == depth:
            break

        elif layer_name == "Flatten":
            break
    
    revMAG = list(reversed(MAG))
    revProc = list(reversed(procedures))
    revSwitch = list(reversed(switch_matrices))
    revW = list(reversed(W))

    num_outputs = revMAG[0].shape[0]
    
    deconved_final_results = np.zeros((num_outputs, SRC.shape[0], SRC.shape[1]), dtype=complex)
    
    for ind_out in range(num_outputs):
        deconvMAG = [None]
        deconvMAG[0] = np.zeros((1, revMAG[0].shape[1], revMAG[0].shape[2]))
        deconvMAG[0][0, :, :] = revMAG[0][ind_out, :, :]

        revSwitch_to_use = [None]*len(revSwitch)
        revSwitch_to_use[0] = np.zeros((1, revSwitch[0].shape[1], revSwitch[0].shape[2]))
        revSwitch_to_use[0][0, :, :] = revSwitch[0][ind_out, :, :]
        revSwitch_to_use[1:] = revSwitch[1:]

        revW_to_use = [None] * len(revW)
        revW_to_use[0] = np.zeros((1, revW[0].shape[1], revW[0].shape[2], revW[0].shape[3]))
        revW_to_use[0][0,:,:,:] = revW[0][ind_out, :, :, :]
        revW_to_use[1:] = revW[1:]

        print('-------feed-back- %d --' % ind_out)
        unpool_ind = 0
        deconv_ind = 0
        for proc_ind, proc_name in enumerate(revProc):
            if proc_name == 'MP':
                deconvMAG.append(relu(get_unpooling2d(images=deconvMAG[proc_ind], switches=revSwitch_to_use[unpool_ind])))
                unpool_ind += 1

            elif proc_name == "conv":
                deconvMAG.append(get_deconvolve(images=deconvMAG[proc_ind], weights=revW_to_use[deconv_ind]))
                deconv_ind += 1

        deconved_final_results[ind_out, :, :] = deconvMAG[-1][:,:,:]
    
    return deconved_final_results

def load_weights():
    import h5py
    model_name = "vggnet5"
    keras_filename = "vggnet5_local_keras_model_CNN_stft_11_frame_173_freq_257_folding_0_best.keras"
    
    print('--- load model ---')
    
    W = []
    with h5py.File(keras_filename, 'r') as f:
        for idx in range(f.attrs['nb_layers']):
            key = 'layer_{}'.format(idx)
            if key in f and 'param_0' in f[key]:
                W.append(f[key]['param_0'][:])
            if len(W) == 5:
                break
    layer_names = ['Convolution2D', 'MaxPooling2D'] * 5 + ['Flatten']
    
    return W, layer_names