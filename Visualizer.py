# This is a data visualization tools script
import numpy as np
import matplotlib.pyplot as plt
import h5py,os

'''
#Path  e.g.  'D:\Code\fcgcn-master\Result_model\Leave_one_session_out\history' in Windows
history('<your result folder path>', 'acc')
# the image will be saved in current folder
'''

def show_data(data, label, channel = 4):
    for i in range(channel):
        index_plot = 100 * (channel + 1) + 10 + (i + 1)
        plt.subplot(index_plot)
        plt.plot(data[i,:])
        if i == 3:
            plt.subplot(index_plot + 1)
            plt.plot(label)

def show_filter_bank(data,index_channel):
    for i in range(9):
        index_plot = 911 + i
        plt.subplot(index_plot)
        plt.plot(data[i, index_channel, :])
        
        
def show_history(path,type_diagram):
    history = h5py.File(path, 'r')
    loss = history['loss']
    acc = history['acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    if type_diagram == 'acc':
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    elif type_diagram == 'loss':
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        plt.show()
def history(path, type_diagram):
    filename_list = []
    if path:
        for root, dirs, files in os.walk(path):
            for filename in files:
                filename_list.append(filename)
                print(filename)
    else :
        print('Please set the path first')
    print('**************Name loaded successfully!**************')
    for i in range(len(filename_list)):
        file = path +'/' + filename_list[i]
        print(file)
        history = h5py.File(file, 'r')
        #loss = history['loss']
        acc = history['acc']
        #val_loss = history['val_loss']
        val_acc = history['val_acc']
        if type_diagram == 'acc':
            plt.plot(acc)
            plt.plot(val_acc)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(filename_list[i]+'_acc.png')
            plt.clf()
        elif type_diagram == 'loss':
            plt.plot(loss)
            plt.plot(val_loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='lower left')
            plt.savefig(filename_list[i]+'_loss.png')
            plt.clf()
    print('***************Data loaded successfully!***************')