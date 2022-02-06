from sklearn import datasets  # 导入库
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch import optim
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def get_data_sklearn():
    # 旧的波士顿房价获取方式，使用会警告
    # 数据需要归一化，范围太大
    boston = datasets.load_boston()  # 导入波士顿房价数据

    train = boston.data  # sample
    target = boston.target  # target
    # 切割数据样本集合测试集
    x_train, x_test, y_train, y_true = train_test_split(train, target, test_size=0.2)  # 20%测试集；80%训练集

    # 归一化
    for i in range(x_train.shape[1]):
        x_train[:, i] = nomorlization(x_train[:, i])
        x_test[:, i] = nomorlization(x_test[:, i])
    y_train = nomorlization(y_train)
    y_true = nomorlization(y_true)
    return x_train, x_test, y_train, y_true  # (404,13)


def get_data_sklearn2():
    # 新的波士顿房价获取方式、分出训练集、测试集、归一化
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    x_train, x_test, y_train, y_true = train_test_split(data, target, test_size=0.2)  # 20%测试集；80%训练集

    # 归一化
    for i in range(x_train.shape[1]):
        x_train[:, i] = nomorlization(x_train[:, i])
        x_test[:, i] = nomorlization(x_test[:, i])
    y_train = nomorlization(y_train)
    y_true = nomorlization(y_true)
    return x_train, x_test, y_train, y_true  # (404,13)


def nomorlization(input):
    # 归一化函数
    mean = np.mean(input)
    # sigma = np.std(input, axis=0)
    max = np.max(input)
    min = np.min(input)
    return (input - mean) / (max - min)


def data_deal(x_train, y_train):
    # 将原本的时间序列分成（13个变量）10个一组，对应10个预测值

    data_pair = []
    for i in range(len(x_train)-20):
        input_ten = x_train[i:i+10]
        target_ten = y_train[i:i+10]
        data_pair.append([input_ten, target_ten])

    # data_pair.pop(-1)
    return data_pair


class time_serise_dataset(Dataset):
    # pytorch的数据集制作方法，方便DataLoader取数，这样可以取batch——size训练
    # 输出为item位置的10个x变量，10个y变量
    def __init__(self, train=True):
        x_train, x_test, y_train, y_true = get_data_sklearn2()
        if train:
            self.data_pair = data_deal(x_train=x_train, y_train=y_train)
        else:
            self.data_pair = data_deal(x_train=x_test, y_train=y_true)

    def __getitem__(self, item):
        x = self.data_pair[item][0]
        y = self.data_pair[item][1]

        return x, y

    def __len__(self):
        return len(self.data_pair)


class Rnn_encoder(nn.Module):
    # rnn编码结构，x为13个变量，隐藏层与解码输入层一致
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=13,
            hidden_size=24,
            num_layers=1,
            batch_first=True,
        )
        self.gru = nn.GRU(input_size=13, hidden_size=13)

    def forward(self, input, hidden):  # hidden--(h_n, h_c) h_0,shape=(num_layers*num_directions,batch_size,hidden_size)
        """

        :param input:
        :param hidden:
        :return:
        r_out :
        h_n : (, , 16)
        """
        #print(input.size(), hidden[1].size())
        r_out, (h_n, h_c) = self.lstm(input, hidden)

        return r_out, (h_n, h_c)


class Rnn_decoder(nn.Module):
    # 解码层，包括全连接层，输出预测值
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=24,
            hidden_size=13,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(13, 1)

    def forward(self, input, hidden):
        r_out, hidden = self.lstm1(input, hidden)
        # print(r_out.shape)
        y = self.fc1(r_out)  # 查看r_out维度

        return y, hidden


def train_one_step(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden1 = torch.zeros(1, 32, 13)
    encoder_hidden2 = torch.zeros(1, 32, 13)
    # print(encoder_hidden.size())
    encoder_hidden = (encoder_hidden1, encoder_hidden2)
    #encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = 10  # 10个数为编码输入次数
    target_len = 10
    # y = torch.Tensor(y)
    encoder_outs = torch.zeros(32, 10, 13)
    loss = 0

    # 编码
    size = len(train_loader.dataset)
    for item, (x, y) in enumerate(train_loader):
        #x = torch.from_numpy(x)
        x = x.to(torch.float32)
        for ei in range(input_len):

            encoder_hidden1 = encoder_hidden1.to(torch.float32)
            encoder_hidden2 = encoder_hidden2.to(torch.float32)
            # encoder_hidden2 = encoder_hidden2.double()
            #print(encoder_hidden1.dtype, encoder_hidden2.dtype)
            # print(encoder_out.dtype)
            #print(x.dtype)

            encoder_out, encoder_hidden = encoder(x, encoder_hidden)
            encoder_outs[:, ei, :] = encoder_out[0][0]

    # for ei in range(input_len):
    #
    #     input = torch.Tensor(x[ei])
    #     input = torch.unsqueeze(input, dim=0)
    #     input = torch.unsqueeze(input, dim=0)
    #     # print(input.shape)
    #     # print(encoder_hidden1.size())
    #     encoder_out, encoder_hidden = encoder(input, (encoder_hidden1, encoder_hidden2))
    #
    #     encoder_outs[:, ei, :] = encoder_out[0][0]

    # decoder_input = torch.tensor([[0]])
        decoder_input = encoder_out
        decoder_hidden = encoder_hidden

        use_teach_forcing = True
        #print(decoder_input.shape, decoder_hidden[0].shape)
        loss_list = []
        # 解码
        if use_teach_forcing:
            for di in range(target_len):
                y2 = y[:, di]

                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                #print(decoder_input.shape)
                # print(decoder_output, target1.size())
                # print(decoder_output.dtype, target1.dtype)
                # target1 = target1.long()\
                #print(decoder_output.dtype,y[:,di])
                #print(y.shape)
                y2 = y2.long()
                decoder_output = decoder_output.float()
                y2 = torch.unsqueeze(y2, dim=0)
                #print(y2.T.shape)
                #y[:, di] = y[:,di].view(32,1)
                # y[:,di] = y[:,di,None]
                loss_list.append(criterion(decoder_output, y2.T))

                # print(y[:,di])

                decoder_input = decoder_output
                #print(decoder_input.shape)
                a = torch.zeros(32, 10, 13)
                #print(a[:,:,0].shape)
                for i in range(13):
                    a[:,:,:] = decoder_input

                decoder_input = a


        else:
            for di in range(target_len):
                decoder_output = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)  # 沿给定dim维度返回输入张量input中 k 个最大值
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, y[di])

        for i in range(len(loss_list)):
            a += loss_list[i]
        # autograd.grad(out)
        # autograd.grad()
        loss = a
        loss.backward(retain_graph=True)
        #encoder_optimizer.step()
        decoder_optimizer.step()
    return loss.item() / target_len


def trainiter(train_loader, encoder, decoder, n_iters, learn_rate=0.01):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learn_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learn_rate)
    # data_pair = data_deal(x_train, y_train)
    # training_pairs = [random.choice(data_pair)for i in range(n_iters)]  # 随机选择迭代次数作为训练
    # print(len(training_pairs))

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss

    for iter in tqdm(range(1, n_iters+1)):

        loss = train_one_step(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        if iter % 10 == 0:
            print('step:%d, loss:%f' % (iter, loss))


def run():
    # x_train, x_test, y_train, y_true = get_data_sklearn()
    train_data = time_serise_dataset(train=True)
    test_data = time_serise_dataset(train=False)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32)
    # print(train_data[0][0].shape) #  [0]为训练对,[0][0]为第一项训练集

    encoder = Rnn_encoder()
    decoder = Rnn_decoder()
    trainiter(train_loader, encoder=encoder, decoder=decoder, n_iters=1000, learn_rate=0.001)


if __name__ == '__main__':
    x_train, x_test, y_train, y_true = get_data_sklearn2()
    #
    data_pair = data_deal(x_train, y_train)
    # print(data_pair[-1][1].shape)
    # for i in range(len(data_pair), -1, -1):
    # data_pair.reverse()
    # print(data_pair[0][0].shape)

    train_data = time_serise_dataset(train=True)  # 383zu
    test_data = time_serise_dataset(train=False)  # 81zu
    for item, (x, y) in enumerate(test_data):
        print('---' * 10, end='%d' % item)
        print(y.shape)

    # print(train_data)
    # train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=32)
    # for item, (x, y) in enumerate(train_loader):
    #     print(x.shape)

