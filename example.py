import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import auralise

def save_spectrogram(spectrogram, filename):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(filename)
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
    depths = [5, 4, 3, 2, 1]
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
                save_spectrogram(np.abs(SRC) * np.abs(MASK), path_src + song_id + '_img' + '/' + filename_img_out)

                filename_img_out = 'filter_for_%s_from_depth_%d_feature_%d.png' % (song_id, depth, deconved_feature_ind)
                save_spectrogram(np.abs(MASK), path_src + song_id + '_img' + '/' + filename_img_out)
