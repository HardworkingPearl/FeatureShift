import torch
import time
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn
from Models import *
from pathlib import Path
from EEGDataset import *
from torch.utils.data import DataLoader
from Utils import *
from itertools import combinations
from random import shuffle
import random
import datetime as dt


class TrainModel():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None  # should be (eeg_channel, frequency, time data point)
        self.model = 'AdaptNet'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 2e-3
        self.num_epochs_global = 90
        self.num_epochs_local = 120  # 120
        self.num_epochs_adapt = 300
        self.num_class = 2
        self.batch_size = 32
        self.batch_size_local = 128
        # self.k_fold = 5  # 10

        # Parameters: Model
        self.dropout = 0.5
        self.use_bias = False
        # self.hiden_node = 100
        self.cv_type = "K_fold"

    def load_processed_data(self, data, label):
        # data: (subject x sample x channel x frequency x datapoint) type = numpy.array
        # label: (subject x sample) type = numpy.array

        self.data = np.expand_dims(data, axis=0)
        self.label = np.expand_dims(label, axis=0)

        self.input_shape = self.data[0][0].shape

        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape, self.label.shape))
        return data.shape

    def load_data(self, path):
        # data: (subject x sample x channel x frequency x datapoint) type = numpy.array
        # label: (subject x sample) type = numpy.array
        path = Path(path)
        filename_data = path / 'data_splited.hdf'
        dataset = h5py.File(filename_data, 'r')
        self.data = dataset['data']
        self.label = dataset['label']
        self.input_shape = self.data[0][0].shape

        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape, self.label.shape))
        return self.data.shape

    def domain_adapt(self):
        # X_train: (sample x channel x frequency x datapoint) type = np.array
        # y_train: (sample,) type = np.array
        save_path = os.getcwd()
        if not os.path.exists(save_path + '/Result_model/K_fold/history'):
            os.makedirs(save_path + '/Result_model/K_fold/history')
        # Data dimention: subject * experiment * sample * channal * datapoint
        # Label dimention: subject * experiment * sample
        # Session: experiment[0:2]-session 1; experiment[2:4]-session 2; experiment[4:end]-session 3
        data = self.data
        label = self.label
        shape_data = data.shape
        shape_label = label.shape
        subject = shape_data[0]  # 32
        trials = shape_data[1]  # 40
        sample = shape_data[2]  # 149

        channel = shape_data[3]
        # frequency = shape_data[3]
        datapoint = shape_data[4]
        print("Train:K-Folds \n1)shape of data:" + str(shape_data) + " \n2)shape of label:" + str(shape_label) +
              " \n3)sample:" + str(sample) + " \n4)datapoint:" + str(datapoint) + " \n5)channel:" + str(
            channel))  # 4)session:" + str(session) +" \n
        save_path = Path(os.getcwd()) / Path('Features/')
        #############################################################################################################
        ##################################### Train a global model  #################################################
        #############################################################################################################
        # data_global = data[:3, :, :, :, :]
        # data_global = np.reshape(data_global,
        #                          (data_global.shape[0] * data_global.shape[1] * data_global.shape[2],
        #                           data_global.shape[3], -1))  # use torch.flatten in the forward process
        # label_global = label[:3, :, :].flatten()
        # # To Do: Train a global model
        # global_model = self.train_global(data_global, label_global)
        # global_model_dict = global_model.state_dict()
        # for dict_key in global_model_dict.keys():
        #     if dict_key.startswith("fc1"):
        #         fc_layer = global_model_dict[dict_key]
        # for name, p in global_model.named_parameters():
        #     if name.startswith('fc1'):
        #         p.requireds_grad = False
        # # Store

        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(fc_layer, save_path / "fc_layer-pytorch.pt")

        if os.path.exists(save_path / "fc_layer-pytorch.pt"):
            fc_layer = torch.load(save_path / "fc_layer-pytorch.pt")
        #############################################################################################################
        ################################ Collect input feature fed into fc layer ####################################
        #############################################################################################################
        # data_feature = data[3:, :, :, :, :]
        # del data
        #
        # data_ = []
        # for sub in range(data_feature.shape[0]):
        #     trial_ = []
        #     for trial in range(data_feature.shape[1]):
        #         with torch.no_grad():
        #             temp_feature = torch.tensor(data_feature[sub, trial, :, :, :]).to(self.device)
        #             global_model.eval()
        #             _, features = global_model(temp_feature)
        #         trial_.append(features)
        #     data_.append(torch.stack(trial_, dim=0))
        # data_feature = torch.stack(data_, dim=0)
        #
        # # Store
        # save_path = Path(os.getcwd()) / Path('Features/')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(data_feature, save_path / "data_feature-pytorch.pt")

        if os.path.exists(save_path / "data_feature-pytorch.pt"):
            data_feature = torch.load(save_path / "data_feature-pytorch.pt")
            del data
        print("shape of features: " + str(data_feature.shape))  # 4)session:" + str(session) +" \n
        #############################################################################################################
        ################################ Fine-tune the fc-layer and collect its weight ##############################
        #############################################################################################################
        data_adapt = data_feature[0:24, :, :, :]
        # data_adapt is data[3:,....] so corresponding
        # labels should also start from  3
        label_adapt = label[3:27, :, :]
        idx = np.arange(20)
        adapt_idx_list = list(combinations(idx, 10))

        samples = 20
        x_adapt = torch.zeros((data_adapt.shape[0], samples, 480))
        z_adapt = torch.zeros((data_adapt.shape[0], samples, 41))
        y_adapt = torch.zeros((data_adapt.shape[0], samples, 480))
        for i in range(data_adapt.shape[0]):  # data_adapt.shape[0]
            random.seed(i)
            idx = adapt_idx_list
            shuffle(idx)
            for j in range(samples):  # 1000
                source_idx = [2 * x for x in idx[j][:-1]] + [2 * x + 1 for x in idx[j][:-1]]
                source_idx.sort()
                target_idx = list(np.arange(40))
                target_idx = [item for item in target_idx if item not in source_idx]

                data_source = data_adapt[i, source_idx, :, :]
                data_source = data_source.flatten(start_dim=0, end_dim=1)
                # data_source = np.reshape(data_source,
                #                          (data_source.shape[0] * data_source.shape[1],
                #                           data_source.shape[2], -1))
                # data_source_mean = data_source.mean(axis=0)
                # data_source_var = data_source.std(axis=0)
                label_source = label_adapt[i, source_idx, :].flatten()

                source_w = self.train_local(fc_layer, data_source, label_source).flatten()

                data_target = data_adapt[i, target_idx, :, :]
                # data_valid = data_target[:,-data_target.shape[1]//5:,:]
                data_target = data_target[:, :-data_target.shape[1] // 5, :]

                # data_target = np.reshape(data_target,
                #                          (data_target.shape[0] * data_target.shape[1],
                #                           data_target.shape[2], -1))
                data_target = data_target.flatten(start_dim=0, end_dim=1)
                # data_valid = data_valid.flatten(start_dim=0, end_dim=1)

                # data_target_mean = data_target.mean(axis=0)
                # data_target_var = data_target.std(axis=0)
                label_target = label_adapt[i, target_idx, :-data_target.shape[1] // 5].flatten()
                label_valid = label_adapt[i, target_idx, -data_target.shape[1] // 5:].flatten()
                target_w = self.train_local(fc_layer, data_target, label_target).flatten()

                # domain_feature = torch.cat([data_source_mean, data_source_var, data_target_mean, data_target_var],
                #                            dim=0)
                x_adapt[i, j, :] = source_w
                # z_adapt[i, j, :] = domain_feature
                z_adapt[i, j, 0] = i
                z_adapt[i, j, 1:19] = torch.tensor(source_idx)
                z_adapt[i, j, 19:41] = torch.tensor(target_idx)
                y_adapt[i, j, :] = target_w

                print('Subject [{}/{}], AdaptSample [{}/{}] '
                      .format(i, data_adapt.shape[0], j, samples))
        x_adapt = x_adapt.flatten(end_dim=1)
        z_adapt = z_adapt.flatten(end_dim=1)
        y_adapt = y_adapt.flatten(end_dim=1)

        # Store
        save_path = Path(os.getcwd()) / Path('Features/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(x_adapt, save_path / "x_tensor-pytorch.pt")
        torch.save(z_adapt, save_path / "z_tensor-pytorch.pt")
        torch.save(y_adapt, save_path / "y_tensor-pytorch.pt")

        # only use mean here
        # z_adapt = z_adapt[:, torch.cat([torch.arange(0, 240), torch.arange(480, 720)])]
        #############################################################################################################
        ##################################### Train a domain adaptation network #####################################
        #############################################################################################################
        if os.path.exists(save_path / "x_tensor-pytorch.pt"):
            x_adapt = torch.load(save_path / "x_tensor-pytorch.pt")
            y_adapt = torch.load(save_path / "y_tensor-pytorch.pt")
            z_adapt = torch.load(save_path / "z_tensor-pytorch.pt")
            print("Torch load-time: {}".format(dt.datetime.now()))

        x_adapt = x_adapt / torch.norm(x_adapt, keepdim=True)
        y_adapt = y_adapt / torch.norm(y_adapt, keepdim=True)

        model_adapt_net = self.train_adaptNet(x_adapt, z_adapt, y_adapt, data_adapt, label_adapt)

        #############################################################################################################
        ##################################### Validate domain adaptation network ####################################
        #############################################################################################################
        data_validate = data_feature[24:, :, :, :]
        label_validate = label[27:, :, :]

        # half for training
        # half: 1/4 for domain mean and variance; half for testing
        # Train and evaluate the model subject by subject
        ACC_old = []
        ACC_new = []
        for i in range(data_validate.shape[0]):
            data_train = data_validate[i, :20, :, :]
            data_feat = data_validate[i, 20:, :sample // 2, :]
            data_test = data_validate[i, 20:, sample // 2:, :]
            data_train = data_train.flatten(start_dim=0, end_dim=1)
            data_feat = data_feat.flatten(start_dim=0, end_dim=1)
            data_test = data_test.flatten(start_dim=0, end_dim=1)
            label_train = label_validate[i, :20, :].flatten()
            label_test = label_validate[i, 20:, sample // 2:].flatten()

            acc_old, acc_new = self.validate_adaptNet(data_train, label_train, data_feat, data_test, label_test,
                                                      model_adapt_net,
                                                      fc_layer)
            print('The average accuracy for the test-subject {} old and new is: {:.4f} vs {:.4f}'.format(i, acc_old,
                                                                                                         acc_new))
            ACC_old.append(acc_old)
            ACC_new.append(acc_new)
        mAcc_old = np.mean(ACC_old)
        mAcc_new = np.mean(ACC_new)
        # Save result here
        # print("Subject:" + str(i) + "\nmACC: %.2f" % mAcc)
        # file = open("result_k_fold.txt", 'a')
        # file.write("\n" + str(datetime.datetime.now()) + '\nMeanACC:' + str(np.mean(ACC_mean)) + ' Std:' + str(
        #     np.std(ACC_mean)) + '\n')
        # file.close()

        return mAcc_old, mAcc_new
        # return 0, 0

    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat, _ = model(x)
            pred = yhat.max(1)[1]  # .type(torch.FloatTensor).to(self.device)
            correct = (pred == y.type(torch.long)).sum()
            acc = correct.item() / len(pred)
            loss = loss_fn(yhat,
                           y.type(torch.long))  # yhat is in one-hot representation; y is scaler dtype = torch.long
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc

        return train_step

    def make_train_step_fc(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]  # .type(torch.FloatTensor).to(self.device)
            correct = (pred == y.type(torch.long)).sum()
            acc = correct.item() / len(pred)
            loss = loss_fn(yhat,
                           y.type(torch.long))  # yhat is in one-hot representation; y is scaler dtype = torch.long
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lss = loss.item()
            del yhat, loss
            return lss, acc

        return train_step

    def make_train_step_adapt(self, model, loss_fn, optimizer):
        def train_step(x, y, data_source, data_target, data_valid, label_valid):
            model.train()
            yhat = model(x, z)
            loss = 1e7 * loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()

        return train_step

    def train_global(self, train_data, train_label):

        print('Avaliable device:' + self.device)
        torch.manual_seed(42)

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs_global

        model = EEGNet().cuda()
        # train_step = optim.Adam(model.parameters(), lr=2e-3, eps=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 60, 75], gamma=0.5)

        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data
        dataset_train = EEGDataset(train_data, train_label)

        # Dataloader for training process      no shuffle
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)

        total_step = len(train_loader)
        # Training process

        # Train and validation loss
        losses = []
        accs = []

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            scheduler.step()
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                # .format(epoch+1, num_epochs, i+1, total_step, loss, acc))
            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

        # save model and loss
        save_path = Path(os.getcwd())

        return model

    def train_local(self, fc_layer, train_data, train_label):
        print('Avaliable device:' + self.device)
        torch.manual_seed(42)
        np.random.seed(42)
        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs_local
        batch_size = self.batch_size_local
        # give the fc layer a pre-trained initialization
        model = FCLayer().cuda()
        dict_model = model.state_dict()
        for dict_key in dict_model.keys():
            if dict_key.startswith("fc"):
                dict_model[dict_key] = fc_layer
        model.load_state_dict(dict_model)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 100], gamma=0.5)  # , 100
        loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)
        train_step = self.make_train_step_fc(model, loss_fn, optimizer)
        # load the data
        dataset_train = EEGDataset(train_data, train_label)
        # Dataloader for training process      no shuffle
        # train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size_local, shuffle=True)
        # total_step = len(train_loader)
        # Training process
        # Train and validation loss
        # losses = []
        # accs = []
        step = np.ceil(train_data.shape[0] / batch_size).astype(np.int64)
        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            scheduler.step()
            epoch_idx = np.arange(train_data.shape[0])
            np.random.shuffle(epoch_idx)
            epoch_data = train_data[epoch_idx, :]
            epoch_label = train_label[epoch_idx]
            # for i, (x_batch, y_batch) in enumerate(train_loader):
            for i in range(step):
                x_batch = epoch_data[batch_size * i:batch_size * (i + 1), :]
                y_batch = epoch_label[batch_size * i:batch_size * (i + 1)]
                x_batch = x_batch.to(self.device)
                y_batch = torch.tensor(y_batch).to(self.device)
                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                torch.cuda.empty_cache()
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                #       .format(epoch + 1, num_epochs, i + 1, total_step, loss, acc))
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, sum(loss_epoch) / len(loss_epoch), sum(acc_epoch) / len(acc_epoch)))
        # save model and loss
        save_path = Path(os.getcwd())
        for name, p in model.named_parameters():
            if name.startswith('fc1'):
                w = p.data
        # del model, train_step
        # gc.collect()
        # torch.cuda.empty_cache()
        return w

    def train_adaptNet(self, x, z, y, data, label):
        torch.manual_seed(42)
        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs_adapt

        model = AdaptNet().cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 100], gamma=0.5)

        loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step_adapt(model, loss_fn, optimizer)
        # load the data
        dataset_train = AdaptDataset(x, z, y, data, label)

        # Dataloader for training process      no shuffle
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)
        total_step = len(train_loader)
        losses = []
        for epoch in range(num_epochs):
            loss_epoch = []
            scheduler.step()
            for i, (x_batch, y_batch, data_source, data_target, data_valid, label_valid) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                data_source = data_source.to(self.device)
                data_target = data_target.to(self.device)
                data_valid = data_valid.to(self.device)
                label_valid = label_valid.to(self.device)
                loss = train_step(x_batch, y_batch, data_source, data_target, data_valid, label_valid)
                loss_epoch.append(loss)
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #       .format(epoch + 1, num_epochs, i + 1, total_step, loss))
            losses.append(sum(loss_epoch) / len(loss_epoch))
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1]))
        return model

    def validate_adaptNet(self, data_train, label_train, data_feat, data_test, label_test, model_adapt_net, fc_layer):
        print('Avaliable device:' + self.device)
        torch.manual_seed(42)

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs_local

        # give the fc layer a pre-trained initialization
        model = FCLayer().cuda()
        dict_model = model.state_dict()
        for dict_key in dict_model.keys():
            if dict_key.startswith("fc"):
                dict_model[dict_key] = fc_layer
        model.load_state_dict(dict_model)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate, eps=1e-4)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50], gamma=0.5)

        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step_fc(model, loss_fn, optimizer)

        # load the data
        dataset_train = EEGDataset(data_train, label_train)
        dataset_test = EEGDataset(data_test, label_test)

        # Dataloader for training process      no shuffle
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)

        total_step = len(train_loader)
        # Training process
        losses = []
        accs = []

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            scheduler.step()
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                # .format(epoch+1, num_epochs, i+1, total_step, loss, acc))
            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

        for name, p in model.named_parameters():
            if name.startswith('fc1'):
                w = p.data.flatten().unsqueeze(dim=0)
                w = w / torch.norm(w, keepdim=True)

        data_train_mean = data_train.mean(axis=0)
        data_train_var = data_train.std(axis=0)

        data_feat_mean = data_feat.mean(axis=0)
        data_feat_var = data_feat.std(axis=0)
        z = torch.cat([data_train_mean, data_train_var, data_feat_mean, data_feat_var]).unsqueeze(dim=0)  #
        with torch.no_grad():
            model_adapt_net.eval()
            w_new = model_adapt_net(w, z).view(2, -1)

        model_new = FCLayer().cuda()
        dict_model = model_new.state_dict()
        for dict_key in dict_model.keys():
            if dict_key.startswith("fc"):
                dict_model[dict_key] = w_new
        model_new.load_state_dict(dict_model)

        if torch.cuda.is_available():
            model_new = model_new.to(self.device)

        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=True)
        test_losses_old = []
        test_acc_old = []

        test_losses_new = []
        test_acc_new = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                model.eval()
                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test.type(torch.long)).sum()
                acc = correct.item() / len(pred)
                val_loss = loss_fn(yhat, y_test.type(torch.long))
                test_losses_old.append(val_loss.item())
                test_acc_old.append(acc)

                model_new.eval()
                yhat = model_new(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test.type(torch.long)).sum()
                acc = correct.item() / len(pred)
                val_loss = loss_fn(yhat, y_test.type(torch.long))
                test_losses_new.append(val_loss.item())
                test_acc_new.append(acc)
        print("acc of old model: {:.4f}\n acc of new model: {:.4f}".format(np.mean(test_acc_old), np.mean(test_acc_new)))
        return np.mean(test_acc_old), np.mean(test_acc_new)
