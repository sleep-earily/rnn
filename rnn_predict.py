import os
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# from torch.autograd import Variable  # torch 中 Variable 模块
import torch.nn as nn
import torch
from prediction_xulie import time_serise_dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_data_time():
    train_data = time_serise_dataset(train=True)
    test_data = time_serise_dataset(train=False)
    print(train_data[-1][0].shape)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)  # torch.Size([32, 10, 13])
    test_loader = DataLoader(dataset=test_data, batch_size=32)

    return train_loader, test_loader


class rnn_time(nn.Module):
    def __init__(self):
        super(rnn_time, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=13,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # print(x.shape)
        r_out, (h_n, h_c) = self.lstm1(x, None)
        y = self.fc1(r_out)
        y = self.fc2(y)  # y---torch.Size([16, 10, 1])
        # print(y.shape)
        return y


def rnn_train(dataloader, rnn, loss_fn, grads):
    # print(len(dataloader.dataset))
    rnn.train()  # 启用 BatchNormalization 和 Dropout

    size = len(dataloader.dataset)

    for item, (X, y) in enumerate(dataloader):
        # print(X.shape)
        x = X.view(-1, 10, 13)
        # print(y[:, 0].shape)
        # print(y[:, 0])
        # print(item)

        x = x.float()
        y = y.float()

        y_temp = y[:, 0].unsqueeze(1)

        pred = rnn(x)  # torch.Size([32, 10, 1])

        loss = loss_fn(pred[:, -1, :], y_temp)  # 只预测一个
        # print(pred[:, -1, :].shape, y_temp.shape)

        grads.zero_grad()
        loss.backward()
        grads.step()


def train_acc(dataloader, net, loss_fn):
    # cnn_net.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.view(-1, 10, 13)
            X = X.float()
            pred = net(X)

            y_temp = y[:, 0].unsqueeze(1)

            pred_y = pred[:, -1, :]
            # print(y_temp)
            # print(pred_y)
            # pred_y = torch.max(pred, 1)[1].data.squeeze()  # torch.max(input,dim)返回每一行中的最大值的标签

            #     accuracy = (pred_y == y).numpy().sum() / y.size(0)
            test_loss += loss_fn(pred_y, y_temp).item()
            correct += (pred_y == y_temp).type(torch.float).sum().item()
            print('预测值：%f, 真实值：%f' % (pred_y[0], y_temp[0]))

    test_loss /= size
    correct /= size
    print(f"train Error: \n train_Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100 * correct


def rnn_test(dataloader, rnn, loss_fn):
    rnn.eval()
    size = len(dataloader.dataset)
    test_loss, correct, accuracy = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            print(1)
            X = X.view(-1, 10, 13)
            X = X.float()
            y_temp = y[:, 0].unsqueeze(1)
            test_out = rnn(X)  # test_out   torch.Size([32, 36])  32--32个batchsize样例，36--各结果可能性
            pred_y = test_out[:, -1, :]  # torch.max(input,dim)返回每一行中的最大值的标签 dim=1每行最大值
            test_loss += loss_fn(pred_y, y_temp).item()
            # correct += (pred_y == y).type(torch.float).sum().item()
            # accuracy += (pred_y == y).numpy().sum()
        test_loss /= size

        # print('| train loss: {} | test accuracy: {} '.format(test_out, accuracy))
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learn_rate = 0.0005
    loss_list, acc_list = [], []
    train_loader, test_loader = get_data_time()

    load_model = True
    if load_model:

        rnn_net = rnn_time()
        rnn_net.load_state_dict(torch.load('RNN_model/rnn_time.pth'))
    else:
        rnn_net = rnn_time()
    # loss_fn = nn.CrossEntropyLoss()  # y只能是分类标签
    loss_fn = nn.MSELoss()
    grads = torch.optim.Adam(rnn_net.parameters(), lr=learn_rate)
    # rnn_train(train_loader, rnn=rnn_net, loss_fn=nn.CrossEntropyLoss, grads=grads)

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        rnn_train(train_loader, rnn=rnn_net, loss_fn=loss_fn, grads=grads)
        train_loss, acc = train_acc(train_loader, rnn_net, loss_fn=loss_fn)
        loss_list.append(train_loss)
        acc_list.append(acc)
        # rnn_test(data_test, rnn2, loss_fn)

    print("Done!")
    #  保存模型
    torch.save(rnn_net.state_dict(), 'RNN_model/rnn_time.pth')
    print("Saved PyTorch Model State to rnn_time.pth")
    print("Done!")
    print()

    plt.subplot(2, 1, 1)
    plt.title('loss')
    plt.plot(loss_list, 'r')
    plt.subplot(2, 1, 2)
    plt.title('acc')
    plt.plot(acc_list, 'b')
    plt.show()


def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learn_rate = 0.0005
    loss_list, acc_list = [], []
    train_loader, test_loader = get_data_time()

    loss_fn = nn.MSELoss()
    # grads = torch.optim.Adam(rnn_net.parameters(), lr=learn_rate)

    rnn_net = rnn_time()
    rnn_net.load_state_dict(torch.load('RNN_model/rnn_time.pth'))

    rnn_test(test_loader, rnn=rnn_net, loss_fn=loss_fn)
    # print('测试完成')


def pred():
    # 10预测1, teach_focing
    count = 0
    # train_loader, test_loader = get_data_time()
    test_data = time_serise_dataset(train=False)
    rnn_net = rnn_time()
    rnn_net.load_state_dict(torch.load('RNN_model/rnn_time.pth'))

    pred_list, target_list = [], []
    for num in range(len(test_data)):
        x, y = test_data[num][0], test_data[num][1]
        with torch.no_grad():
            # print(x[np.newaxis, :, :].shape)
            # x = x.view(-1, 10, 13)

            x = x[np.newaxis, :, :]

            x = torch.tensor(x, dtype=torch.float32)
            print(x.dtype, y.shape)
            pred2 = rnn_net(x)
            pred_temp = pred2[:, -1, :]
            y_temp = y[0]
            pred_list.append(pred_temp.item())
            target_list.append(y_temp)
            print('预测值：%f, 真实值：%f' % (pred_temp[0], y_temp))

    return pred_list, target_list


def pred_steps():
    count = 0
    # train_loader, test_loader = get_data_time()
    test_data = time_serise_dataset(train=True)
    rnn_net = rnn_time()
    rnn_net.load_state_dict(torch.load('RNN_model/rnn_time.pth'))

    pred_list, target_list = [], []
    for num in range(len(test_data)):
        x, y = test_data[num][0], test_data[num][1]
        with torch.no_grad():
            # print(x[np.newaxis, :, :].shape)
            # x = x.view(-1, 10, 13)

            x = x[np.newaxis, :, :]
            x = torch.tensor(x, dtype=torch.float32)

            pred2 = rnn_net(x)
            pred_temp = pred2[:, -1, :]
            y_temp = y[0]
            print(num)
            # print(test_data[num+1][0][-1])
            # test_data[num + 1][0][-1] = y_temp
            # print(test_data[num + 1][0])
            pred_list.append(pred_temp)
            target_list.append(y_temp)
            # print('预测值：%f, 真实值：%f' % (pred_temp[0], y_temp))

    return pred_list, target_list


if __name__ == '__main__':
    # model()
    # test_model()

    pred_list, target_list = pred()
    plt.plot(pred_list, c='b')
    plt.plot(target_list, c='r')
    plt.show()
    # pred_steps()  # 单步预测