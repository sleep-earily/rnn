# rnn
对rnn的初始学习部分

prediction_xulie提供数据处理工作，将sklearn提取的波士顿房价数据下载、归一化、分组（10个为一组）、制作训练集测试集。

rnn_predict为rnn结构，x为13维变量，10个x预测1个y

补充：原本打算在prediction_xulie上用transformer,但是失败了，因此先用rnn测试
