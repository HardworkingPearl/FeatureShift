import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # for support set
        self.fc11 = nn.Linear(16 * 15, 128)
        self.fc12 = nn.Linear(128, 64)

        # for support set mean
        self.fc21 = nn.Linear(16 * 15, 128)
        self.fc22 = nn.Linear(128, 64)

    def forward(self, x):
        """

        @param x: size (batch, support_size, dim)
        x = torch.zeros((32, 1000, 240))
        mean = torch.zeros((32, 240))
        @return:
        """
        batch = x.shape[0]
        # support = x.shape[1]
        dim = x.shape[2]

        x = x.flatten(end_dim=1)
        x = self.fc11(x)
        x = self.fc12(x)
        x = x.view((batch, -1, dim))

        mean = x.mean(dim=1)
        mean = self.fc21(mean)
        mean = self.fc22(mean)
        # mean (batch, 1, 1, dim)
        mean = mean.unsqueeze(dim=1).unsqueeze(dim=2)
        # x (batch, 1000, dim, 1)
        # att_score (batch, support, 1, 1)
        att_score = F.softmax(torch.matmul(mean, x.unsqueeze(dim=3)), dim=-1).squeeze(dim=2)
        att_score = torch.add(att_score, torch.zeros((att_score.shape[0], att_score.shape[1], x.shape[2])))
        output = x * att_score
        output = output.sum(dim=1)

        return output

class TrainProcess(nn.Module):
    def __init__(self):
        super(TrainProcess, self).__init__()
        self.adapt_net = AdaptNet()
        self.embedding_net = EmbeddingNet()
        self.classifer = FCLayer()

    def forward(self, x_batch, y_batch, data_source, data_target, data_valid, label_valid):


        for dict_key in dict_model.keys():
            if dict_key.startswith("fc"):
                dict_model[dict_key] = w_new
        model_new.load_state_dict(dict_model)

class AdaptNet(nn.Module):
    """
    network to adapt the parameters from one domain to another
    """

    def __init__(self):
        super(AdaptNet, self).__init__()
        # the input embedding would be (16, 1, 15) *4
        # two domain, variance and mean
        # domain feature comparison net
        self.fc11 = nn.Linear(64*2, 128)  # try only mean 2*
        self.fc12 = nn.Linear(128, 64)
        # parameter processing
        self.fc2 = nn.Linear(16 * 15 * 2, 128)
        self.fc3 = nn.Linear(128 + 64, 16 * 15 * 2)

    def forward(self, x, z):
        x_0 = x
        z = F.relu(self.fc11(z))
        z = F.relu(self.fc12(z))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, z), 1)
        x = self.fc3(x)
        # short cut
        x = x + x_0
        return x


class FCLayer(nn.Module):
    def __init__(self):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(16 * 1 * 15, 2, bias=False)

    def forward(self, x):
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class EEGNet(nn.Module):
    def __init__(self, Chans=32,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.T = 512

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)  # along the axis = 1, C = F1
        self.depthconv1 = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 4), padding=0, groups=F1,
                                    bias=False)

        # Layer 2
        self.batchnorm2 = nn.BatchNorm2d(D * F1, False)
        self.pooling2 = nn.AvgPool2d(1, 5)
        self.dropout2 = nn.Dropout(p=dropoutRate)

        # Layer 3
        self.separableconv3 = SeparableConv2d(F2, F2, kernel_size=(1, 16))
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.pooling3 = nn.AvgPool2d((1, 5))
        self.dropout3 = nn.Dropout(p=dropoutRate)
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 512 timepoints.
        self.fc1 = nn.Linear(16 * 1 * 15, 2, bias=False)

    def forward(self, x):  # , w
        # Datatype - float32(both
        # X and Y)
        # X.shape - (  # samples, 1, #timepoints, #channels)
        # Y.shape - (  # samples)

        # Layer 1
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthconv1(x)

        # Layer 2
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.separableconv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # FC Layer
        x = x.view(-1, 16 * 1 * 15)

        features = x
        # x = F.sigmoid(F.linear(x, w))
        x = F.log_softmax(self.fc1(x), dim=1)
        return x, features
