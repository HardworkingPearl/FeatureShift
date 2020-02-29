# This is a script to do pre-processing on the EEG data
import numpy as np
import math
import h5py
import os
import Preprocessing as PreP
from Utils import *


class DataFormat:
    def __init__(self):
        self.data = None
        self.label = None
        self.data_processed = None
        self.label_processed = None

    def load_data(self, data, label):
        self.data = data
        self.label = label
        print("The data and label are loaded successfully!")

    def load_processed_data(self, data, label):
        self.data_processed = data
        self.label_processed = label
        print("The data and label are loaded successfully!")

    def remove_artifact(self, low_frequency=4, high_frequency=45, sampling_rate=256, filter_order=5):
        for i in range(len(self.data)):
            # self.data[i] = PreP.band_pass(self.data[i], low_frequency, high_frequency, sampling_rate, filter_order)
            # self.data[i] = PreP.ICAartiRemove(self.data[i], sampling_rate) # Using ICA to remove eye actitity
            self.data[i] = PreP.ArtifactRemoval(self.data[i])
        print("Artifacts removed by band-pass filter")

    def divide_data(self, number_experiment=6, save=0):
        # Variables
        data_list = []
        label_list = []
        data_array = None
        label_array = None

        # TODO: data formatting sublject by subject
        for sub in range(len(self.data)):
            # use temp variables
            data_subject_experiment = []
            label_subject_experiment = []
            # Use this index to divide the data into different experiment
            # Iindex_experiment = [0]
            data_per_subject = self.data[sub]
            label_per_subject = self.label[sub]
            size = data_per_subject.shape
            channel = size[1]  # DEAP dataset channel is in the second channel
            print("Subject:" + str(sub) + " The data has " + str(channel) + " channels")

            data_list.append(np.stack(data_per_subject, axis=0))
            label_list.append(np.stack(label_per_subject, axis=0))

        data_array = np.stack(data_list, axis=0)
        label_array = np.stack(label_list, axis=0)
        data_array = np.expand_dims(data_array, axis=4)
        # data_array = data_per_subject
        # label_array = label_per_subject
        print(data_array.shape)
        print(label_array.shape)
        # change the label representation 1.0 -> 0.0; 2.0 -> 1.0

        label_array[label_array < 5] = 0.0
        label_array[label_array >= 5] = 1.0

        self.data_processed = data_array
        self.label_processed = label_array

        print("The data is divided into different experiment for the Leave One Session Out CV")
        print("The data shape is:" + str(self.data_processed.shape) + "The label shape is:" + str(
            self.label_processed.shape))
        # TODO: Save the processed data here
        if save == 1:
            if self.data_processed.all() != None:
                save_path = os.getcwd()
                upper_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
                if not os.path.exists(upper_dir + '/Data_processed'):
                    os.makedirs(upper_dir + '/Data_processed')
                filename_data = upper_dir + '/Data_processed/data_different_experiment.hdf'
                save_data = h5py.File(filename_data, 'w')
                save_data['data'] = self.data_processed
                save_data['label'] = self.label_processed
                save_data.close()
                print("Data and Label saved successfully! at:" + filename_data)
            else:
                print("data_processed is None")

    def divide_filter_bank(self, sampling_rate, filter_order, save):
        self.data_processed = PreP.filter_bank(self.data_processed, sampling_rate, filter_order)
        print('DataFormat: divide filter bank successfully! The new data size is:' + str(self.data_processed.shape))
        # TODO: Save the processed data here
        if save == 1:
            # if self.data_processed.all() != None: # process forever
            save_path = os.getcwd()
            upper_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            if not os.path.exists(upper_dir + '/Data_processed'):
                os.makedirs(upper_dir + '/Data_processed')
            for sub in range(self.data_processed.shape[0]):
                filename_data = upper_dir + '/Data_processed/data_subject_' + str(sub) + '.hdf'
                save_data = h5py.File(filename_data, 'w')
                save_data['data'] = self.data_processed[sub, :, :, :, :]
                save_data['label'] = self.label_processed[sub, :]
                save_data.close()
                print("Data and Label saved successfully! at:" + filename_data)

    #            else :
    #              print("data_processed is None")

    def extract_PSD(self):
        self.data_processed = PreP.PSD(self.data_processed)
        print('DataFormat: extract the PSD successfully! The new data size is:' + str(self.data_processed.shape))

    def relative_power(self):
        for sample in range(self.data_processed.shape[0]):
            self.data_processed[sample, :, :, :] = relative_power(self.data_processed[sample, :, :, :])

    def split_data(self, segment_length=1, overlap=0, sampling_rate=256, save=0, sub=0):

        # Parameters
        ## to save memory, convert from float 64 to float 16
        data = self.data_processed
        label = self.label_processed
        # Split the data given
        data_shape = data.shape
        label_shape = label.shape
        data_overlap = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_splited = []
        label_splited = []

        # label = np.expand_dims(label, axis = 2)
        number_segment = int((data_shape[-1] - data_segment) // (data_overlap))

        for trial in range(data.shape[1]):
            data_trial=[]
            label_trial=[]
            for i in range(number_segment + 1):
                data_sample = data[:, trial, :, int(i * data_overlap):int(i * data_overlap + data_segment)]
                # data_sample= np.expand_dims(data_sample, axis = 0)
                # data_splited.append(data_sample)
                # label_splited.append(label[:, trial])
                data_trial.append(data_sample)
                label_trial.append(label[:, trial])
            # TO DO: one trials' samples stacking together
            data_splited.append(np.stack(data_trial, axis=1))
            label_splited.append(np.stack(label_trial, axis=1))
        data_splited = np.stack(data_splited, axis=1)
        label_splited = np.stack(label_splited, axis=1)
        if save == 1:
            save_path = os.getcwd()
            upper_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            if not os.path.exists(upper_dir + '/Data_forTrain'):
                os.makedirs(upper_dir + '/Data_forTrain')
            filename_data = upper_dir + '/Data_forTrain/data_splited.hdf' #data_subject_' + str(sub) + '.hdf'
            save_data = h5py.File(filename_data, 'w')
            save_data['data'] = data_splited  # self.data_processed
            save_data['label'] = label_splited  # self.label_processed
            save_data.close()
            print("Data and Label saved successfully! at:" + filename_data)

            # sub_data = np.squeeze(sub_data, axis = 0)

        # data_splited_array = np.dstack(data_splited[i] for i in range(len(data_splited)))
        # data_splited_array = np.transpose(train_data,(0,2,1,3,4))

        print("The data and label are splited: Data shape:" + str(data_splited.shape) + " Label:" + str(
            label_splited.shape))
        self.data_processed = data_splited
        self.label_processed = label_splited
#        #TODO: Save the processed data here
#        if save == 1:
#            if self.data_processed.all() != None:
#              
#              save_path = os.getcwd()
#              filename_data = save_path+'/Data_processed/data_splited.hdf'
#              save_data = h5py.File(filename_data, 'w')
#              save_data['data'] = self.data_processed
#              save_data['label'] = self.label_processed
#              save_data.close()
#              print("Data and Label saved successfully! at:" + filename_data)
#            else :
#              print("data_splited is None")
