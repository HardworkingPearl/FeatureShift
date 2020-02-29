# This is a script to load the data
import scipy.io
import numpy as np
import os
import DataFormat
import _pickle
import h5py


class DataLocalLoader:
    def __init__(self):
        self.path = None
        self.data = []
        self.label = []

    def set_path(self, data_path):
        self.path = data_path
        print('**************Set data path successfully!**************')
        print('The path is: ' + self.path)

    def load_training_data(self, subject):
        data_dictionary = h5py.File(self.path + '/' + 'data_subject_' + str(subject) + '.hdf', 'r')
        data = np.array(data_dictionary.get('data'))  # save more space and memory
        self.data = data  # last 8 channels are not eeg; so exlude 33:40
        labels = np.array(data_dictionary.get('label'))
        self.label = labels
        data_dictionary.close()
        print('The shape of data is:' + str(self.data.shape))
        print('The shape of label is:' + str(self.label.shape))

    def load_processed_data(self, subject):
        filename_list = []
        if self.path:
            for root, dirs, files in os.walk(self.path):
                for filename in files:
                    filename_list.append(filename)
                    print(filename)
        else:
            print('Please set the path first')
        print('**************Name loaded successfully!**************')

        # sort filename_list with number in the name string
        if len(filename_list) > 1:
            filename_list = sorted(filename_list,
                                   key=lambda x: int(x.partition('_')[2].partition('_')[2].partition('.')[0]))
            data_dictionary = h5py.File(self.path + '/' + filename_list[subject], 'r')
            labels = np.array(data_dictionary.get('label'))
            data = np.array(data_dictionary.get('data'))  # save more space and memory
        else:  # data has been splited into 9 filter banks
            data_dictionary = h5py.File(self.path + '/' + filename_list[0], 'r')
            labels = np.array(data_dictionary.get('label'))
            data = np.array(data_dictionary.get('data'))[:, :, :, :, 0]  # save more space and memory
        self.data = data  # last 8 channels are not eeg; so exlude 33:40
        self.label = labels
        data_dictionary.close()
        print('The shape of data is:' + str(self.data.shape))
        print('The shape of label is:' + str(self.label.shape))

    def load(self):
        filename_list = []
        if self.path:
            for root, dirs, files in os.walk(self.path):
                for filename in files:
                    filename_list.append(filename)
                    print(filename)
        else:
            print('Please set the path first')
        print('**************Name loaded successfully!**************')
        # every subject has 40 trials
        idx = np.arange(40)
        np.random.shuffle(idx)

        for i in range(len(filename_list)):
            data_dictionary = _pickle.load(open(self.path + '/' + filename_list[i], 'rb'), encoding='latin1')
            file = open("data_record.txt", 'a')
            file.write(filename_list[i] + '    Subject No.:' + str(i) + '\n')
            file.close()
            data = data_dictionary['data'].astype(np.float32)  # save more space and memory

            self.data.append(data[idx, :32, :])  # last 8 channels are not eeg; so exlude 33:40
            labels = data_dictionary['labels'].astype(np.float32)
            self.label.append(labels[idx, 1])
            print('The shape of data is:' + str(self.data[-1].shape))
            print('The shape of label is:' + str(self.label[-1].shape))
        print('***************Data loaded successfully!***************')


# if __name__ == "__main__":
#     dataloader = DataLoader()
#     data_path = os.getcwd()[:-22]
#     dataloader.set_path(data_path + '/data_mat')
#     dataloader.load()
#     dataformat = DataFormat.DataFormat()
#     dataformat.load_data(dataloader.data, dataloader.label)
#     # high_frequency can not be larger than 128
#     dataformat.remove_artifact(low_frequency=4, high_frequency=45, sampling_rate=256, filter_order=8)
#     dataformat.divide_data(number_experiment=6, save=1)
#     dataformat.split_data(segment_length=4, overlap=0.5, sampling_rate=256, save=1)
