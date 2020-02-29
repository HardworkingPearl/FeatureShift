# This is a program to do pre-processing for the data
# 1 band-pass filter
# 2 PSD
# 3 DE
# 4 DWT

import math
import numpy as np
from scipy import signal
import scipy.sparse as sp

# import mne

'''
def ICAartiRemove(eeg, sampling_rate):

    ch_names = [ "TP9", "TP10", "FP1", "FP2"] 
    info = mne.create_info(ch_names, sampling_rate, ch_types = ["eeg"] *4)
    raw = mne.io.RawArray(eeg, info)               # n_channels, n_times   
    raw_tmp = raw.copy()
    raw_tmp.filter(1,None)
    ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
    ica.fit(raw_tmp)
    ica.exclude = [1]
    raw_corrected = raw.copy()
    ica.apply(raw_corrected)
    print(" ICA analysis finished!")
    return raw_corrected.get_data()
'''


def band_pass(data, low_frequency, high_frequency, sampling_rate, filter_order=8):
    wn1 = 2 * low_frequency / sampling_rate
    wn2 = 2 * high_frequency / sampling_rate
    b, a = signal.butter(filter_order, [wn1, wn2], 'bandpass')
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def DE(data, fs, fbank):
    # data: 1d np.array
    # DE: float scaler
    data = band_pass(data, fbank[0], fbank[1], fs, 5)
    DE = 0.5 * math.log(2 * 3.14 * 2.718 * math.sqrt(np.std(data)))
    return DE


def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def PSD(data):
    data = np.square(data)
    return data


def ArtifactRemoval(data, nComp=14, kCompMat=[22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 48, 52], \
                    kCompMat2=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26]):
    '''
    Smooth data with average values;
    modified for DEAO dataset
    
    many zeros in data after this
    
    '''
    fRefData = np.zeros(data.shape)
    fOutEEGData = np.zeros(data.shape)
    dataLength = data.shape[1]
    for i in range(data.shape[0]):
        for j in range(nComp):
            for k in range(dataLength - kCompMat[j]):
                fRefData[:, k] = np.mean(data[:, k:k + kCompMat[j]], axis=1)
            fOutEEGData[:, :dataLength - kCompMat2[j]] = \
                data[:, kCompMat2[j]:dataLength] - fRefData[:, :dataLength - kCompMat2[j]] + \
                fOutEEGData[:, :dataLength - kCompMat2[j]]
    fOutEEGData = fOutEEGData / nComp
    return fOutEEGData


def filter_bank(data, sampling_rate=256, filter_order=5):
    # filter band covers 4-40Hz signal, with 4 Hz each bank
    #    index_fbank = [[5,8],[9,12],[13,16],[17,20],[21,24],
    #                   [25,28],[29,32],[33,36],[37,40]]
    index_fbank = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
                   [25, 28], [28, 32], [32, 36], [36, 40]]
    Data_fbank = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(index_fbank), data.shape[-1]))
    for i in range(len(index_fbank)):
        Data_fbank.append(band_pass(data, index_fbank[i][0], index_fbank[i][1], sampling_rate, filter_order))
    print('Preprocessing: Divide the data into 9 filter bank successfully!')
    return Data_fbank  # np.concatenate(Data_fbank, axis = 3)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def RP(data, fbank, fs):
    # data: channel x datapoint
    # output: channel x 1 relative power
    f = len(fbank)
    data_fb = []
    for i in range(f):
        index = fbank[i]
        data_fb.append(band_pass(data, index[0], index[1], fs, 5))
    data_fb = np.stack(data_fb, axis=0)
    # frequency x channel x datapoint
    shape = data_fb.shape
    data_channel = []
    for j in range(shape[1]):
        data_temp = data_fb[:, j, :]
        data_temp = np.power(data_temp, 2)
        data_temp = np.sum(data_temp, axis=-1)
        power_sum = np.mean(data_temp)
        data_temp = data_temp / power_sum
        data_channel.append(data_temp)
    data_channel = np.stack(data_channel, axis=0)

    return data_channel
