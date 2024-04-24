import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import auralise
from tensorflow import keras
import random

# Ensure the use of the Agg backend for matplotlib to avoid any need for an X server
plt.switch_backend('Agg')

def save_spectrogram(spectrogram, filename, song_id, dpi=300):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the filename for the spectrogram image
    filename_img_out = filename
    
    # Save the spectrogram using a colormap 'coolwarm'
    plt.figure()
    plt.imshow(np.flipud(librosa.amplitude_to_db(spectrogram, ref=np.max)), cmap='coolwarm', aspect='auto')
    plt.axis('off')  # No axes for a cleaner look

    # Save the high resolution image
    plt.savefig(filename_img_out, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # load learned weights
    W, layer_names = auralise.load_weights()
    num_conv_layer = len(W)

    # load files
    print('--- prepare files ---')
    filenames_src = ['bach.wav', 'dream.wav', 'toy.wav']
    path_SRC = 'src_songs/'
    path_src = 'src_songs/'
    print('--- Please modify above to run on your file ---')
    filenames_SRC = []

    N_FFT = 512
    SAMPLE_RATE = 11025

    # get STFT of the files.
    for filename in filenames_src:
        song_id = filename.split('.')[0]
        if os.path.exists(path_SRC + song_id + '.npy'):
            pass
        else:
            src, _ = librosa.load(os.path.join(path_src, filename), sr=SAMPLE_RATE, mono=True, duration=4.)
            SRC = librosa.stft(src, n_fft=N_FFT, hop_length=N_FFT//2)
            if SRC.shape[1] > 173:
                SRC = SRC[:, :173]
            elif SRC.shape[1] < 173:
                temp = np.zeros((257, 173), dtype=complex)
                temp[:, :SRC.shape[1]] = SRC
                SRC = temp
            np.save(path_SRC + song_id + '.npy', SRC)
        filenames_SRC.append(song_id + '.npy')

    # deconve
    depths = [4, 3, 2, 1]
    for filename in filenames_SRC:
        song_id = filename.split('.')[0]
        SRC = np.load(path_SRC + filename)
        filename_out = '%s_a_original.wav' % (song_id)
        if not os.path.exists(path_src + song_id):
            os.makedirs(path_src + song_id)
        if not os.path.exists(path_src + song_id + '_img'):
            os.makedirs(path_src + song_id + '_img')
        sf.write(path_src + song_id + '/' + filename_out, librosa.istft(SRC, hop_length=N_FFT//2), SAMPLE_RATE, 'PCM_24')

        for depth in depths:
            print('--- deconve! ---')
            deconvedMASKS = auralise.get_deconve_mask(W[:depth], layer_names, SRC, depth)
            print('result; %d masks with size of %s' % (deconvedMASKS.shape[0], str(deconvedMASKS.shape[1:])))
            for deconved_feature_ind, deconvedMASK_here in enumerate(deconvedMASKS):
                MASK = np.zeros(SRC.shape, dtype=complex)
                MASK[:deconvedMASK_here.shape[0], :deconvedMASK_here.shape[1]] = deconvedMASK_here
                deconvedSRC = SRC * MASK
                filename_out = '%s_deconved_from_depth_%d_feature_%d.wav' % (song_id, depth, deconved_feature_ind)
                sf.write(path_src + song_id + '/' + filename_out, librosa.istft(deconvedSRC, hop_length=N_FFT//2), SAMPLE_RATE, 'PCM_24')
                filename_img_out = 'spectrogram_%s_from_depth_%d_feature_%d.png' % (song_id, depth, deconved_feature_ind)
                save_spectrogram(np.abs(SRC) * np.abs(MASK), os.path.join(path_src, song_id + '_img', filename_img_out), song_id)
                filename_img_out = 'filter_for_%s_from_depth_%d_feature_%d.png' % (song_id, depth, deconved_feature_ind)
                save_spectrogram(np.abs(MASK), os.path.join(path_src, song_id + '_img', filename_img_out), song_id)
