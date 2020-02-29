# Main function
from DataLocalLoader import *
from DataFormat import *
# from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from Train import *
import os
import glob
import os

if __name__ == "__main__":
    ############################# Data processing ################################
    # print(os.getcwd())
    # dataloader = DataLocalLoader()
    # upper_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # dataformat = DataFormat()
    # #
    # #
    # dataloader.set_path(upper_dir + '/data_preprocessed_python')
    # dataloader.load()
    #
    # dataformat = DataFormat()
    # dataformat.load_data(dataloader.data, dataloader.label)
    #
    # # high_frequency can not be larger than 128
    # # dataformat.remove_artifact(low_frequency = 0.3, high_frequency = 45, sampling_rate = 256, filter_order = 5)
    #
    # dataformat.divide_data(number_experiment=40, save=1)
    # #
    # #    dataformat.divide_filter_bank(sampling_rate = 128, filter_order = 5, save = 1)
    #
    # subject = 0  # 0.6531994 , 0.57831101, 0.6328497 , 0.44166667, 0.60762649
    # dataloader.set_path(upper_dir + '/Data_processed')
    # dataloader.load_processed_data(subject)
    # dataformat.load_processed_data(dataloader.data, dataloader.label)
    # #    # dataformat.extract_PSD()
    # #    dataformat.relative_power()
    # #    # smaller overlap cuz smaller sampling_rate
    # dataformat.split_data(segment_length=4, overlap=0.9, sampling_rate=128, save=1)

    ############################## train ###################################
    mAcc_list = []
    dataformat = DataFormat()
    dataloader = DataLocalLoader()
    upper_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print(upper_dir)
    # remove last processed subject's data
    path = upper_dir + '/Data_forTrain/*'
    files = glob.glob(path)  # r'path/*')
    # for items in files:
    #     os.remove(items)


    # dataloader.set_path(upper_dir + '/Data_forTrain')
    # dataloader.load_training_data(0)
    train = TrainModel()
    # train.parameter(cv="K_folds", model="EEGNet", channel=32, number_class=2,
    #                     datapoint=512, drop_out=0.5, sampling_rate=128,
    #                     F1=8, D=2, F2=16, normal_rate=0.25, dropout_type='Dropout',
    #                     learning_rate=0.001, epoch=500, batch_size=200)
    # Location: without '\' at the end
    data_shape = train.load_data(upper_dir + '/Data_forTrain/')
    data_shape = [data_shape[1], data_shape[2], data_shape[-1] // 16]
    print("The data shape is: ", data_shape)
    # cv : 1) 'Leave_one_session_out' type = string
    #      2) 'K_fold' type = string
    #      3) 'Leave_one_subject_out' type = string
    # model: Please input 'GCN' type = string
    # train.set_parameter(cv='K_fold',
    #                     model='SVM',
    #                     input_shape=data_shape,
    #                     number_class=2,
    #                     random_seed=42,
    #                     learning_rate=0.001,
    #                     epoch=200,
    #                     batch_size=256,
    #                     k_fold=5,
    #                     max_degree=4,
    #                     units=96,
    #                     dropout_gcn=0.2,
    #                     dropout_fc=0.4,
    #                     use_bias=False,
    #                     hiden_node=512)
    mAcc_o, mAcc_n = train.domain_adapt()
    print('The average accuracy old and new is: {:.4f} vs {:.4f}'.format( mAcc_o, mAcc_n))
    # command = 'poweroff'
    # sudoPassword = '1314518160ZQh[]'
    # str = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
