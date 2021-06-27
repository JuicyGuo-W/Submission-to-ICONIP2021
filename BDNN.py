import torch
import pandas as pd
from torch import nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 5  # batch size
epochs = 50  # epoches
learning_rate = 1e-2  # learning rate

data = pd.read_excel('data/facial-features.xlsx', sheet_name=[2])[2]  # load all data
data.drop(data.columns[0], axis=1, inplace=True)  # drop first column as it is identifier
data = data.sample(frac=1)  # disordered data
data_match = data.loc[data['outputs'] == 1,]  # split match & not match data
data_not_match = data.loc[data['outputs'] == 0,]

# randomly split data into 25 out of 36 training sets and 11 test sets(4 'match' and 7 'not match')
# hyperparameters on split dataset
test_match_num = 4
test_num = 11

# generate match data in test set, the same for the rest
test_match = data_match.sample(
    frac=(test_match_num / len(data_match.index)))  # 4
test_not_match = data_not_match.sample(
    frac=(test_num - test_match_num) / len(data_not_match.index))  # 8
train_match = data_match.drop(labels=test_match.index)  # 12 - 4 = 8 items
train_not_match = data_not_match.drop(
    labels=test_not_match.index)  # 25 - 8 = 17 items

# concatenate to generate train and test datasets
train = pd.concat([train_match, train_not_match])
test = pd.concat([test_match, test_not_match])

n_features = train.shape[1] - 1  # number of features
train_features = train.iloc[:, :-1]  # split training data into input and target
train_output = train.iloc[:, -1]  # the former columns are features, the last one is target
test_features = train.iloc[:, :-1]
test_output = train.iloc[:, -1]

# create Tensors to hold inputs and outputs
X_train = torch.Tensor(train_features.values).float()
Y_train = torch.Tensor(train_output.values).long()
X_test = torch.Tensor(test_features.values).float()
Y_test = torch.Tensor(test_output.values).long()

# DataLoader
train = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train, batch_size=5, shuffle=True)
test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size=5, shuffle=True)

# built model
in_dim = 1 * 182  # input layer
h1 = 30  # hidden layer 1
h2 = 10  # hidden layer 2
out_dim = batch_size  # output layer

# record history to plot
train_loss = []  # training loss
test_loss = []  # testing loss
train_acc = []  # training accuracy
test_acc = []  # testing accuracy


class BDNN_net(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim):  # define a model
        super(BDNN_net, self).__init__()  # inherit nn.Module
        self.linear1 = nn.Linear(in_dim, h1)  # input->hidden1
        self.relu1 = nn.ReLU(True)  # relu
        self.linear2 = nn.Linear(h1, h2)  # hidden1->hidden2
        self.relu2 = nn.ReLU(True)  # relu
        self.linear3 = nn.Linear(h2, out_dim)  # hidden2->output

    def forward(self, x):  # Bidirectional Forward process
        x = self.linear1(x)  # input->hidden1
        x = self.relu1(x)  # relu
        x = self.linear2(x)  # hidden1>hidden2
        x = self.relu2(x)  # relu
        x = self.linear3(x)  # hidden2->output
        return x

    def reverseforward(self, x):  # Bidirectional reverse process
        h = self.relu2(torch.matmul(  # mutiple hidden2->output weights and outputs
            x, self.linear3.weight.type(torch.LongTensor)))
        h = self.relu1(torch.matmul(  # mutiple hidden1->hidden2 weights and outputs
            h, self.linear2.weight.type(torch.LongTensor)))
        return torch.matmul(h, self.linear1.weight.type(torch.LongTensor))


model = BDNN_net(in_dim, h1, h2, out_dim)  # built a model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # set an optimizer


def train(flag):  # training method
    model.train()  # set train mode
    loss_total = 0.0  # record loss for each epoch
    acc = 0.0  # record accuracy for each epoch
    for index, train_data in enumerate(train_loader, 1):  # for each batch
        optimizer.zero_grad()  # set zero gradients
        features, label = train_data  # get features and labels from train loader
        features = features.view(features.size(0), -1)
        label = label.long()  # flatten inputs
        loss_forward = F.cross_entropy(model(features), label)  # calculate loss forward using cross_entropy
        out = model.reverseforward(label)
        out = torch.unsqueeze(out, 0)
        # features=features.long()
        loss_back = F.mse_loss(out, features)  # calculate loss backward using MSE
        print(f'f:{loss_forward}, b:{loss_back}')  # print process
        loss = loss_forward + loss_forward
        print(loss.item())  # calculate loss by the sum of two ways
        loss_total += loss.item()  # calculate loss in each epoch
        _, pred = torch.max(model(features), 1)  # calculate prediction,maximum 1
        acc += (pred == label).float().mean()  # calculate accuracy
        loss.backward()  # back propagation
        optimizer.step()  # update weights
    print(f'Train Loss: {loss_total / index:.6f}, Acc: {acc / index:.6f}')  # print process
    train_loss.append(loss_total / index)  # record loss each epoch
    train_acc.append(float(acc / index))  # record accuracy each epoch


def test(flag):  # testing method
    model.eval()  # test mode
    loss_total = 0.0  # record loss
    acc = 0.0  # record accuracy
    for test in test_loader:  # get data from test_loader
        features, label = test  # get features and labels from test loader
        features = features.view(features.size(0), -1)  # flatten features
        with torch.no_grad():
            out = model(features)  # calculate output
            loss_forward = F.cross_entropy(out, label)  # calculate loss forward
            loss_back = F.mse_loss(model.reverseforward(label), features)  # calculate loss backward
            print(f'f{loss_forward}, b:{loss_back}')  # print process
            loss = loss_forward + loss_forward  # calculate loss by the sum of two ways
        loss_total += loss.item()  # calculate loss for each epoch
        _, pred = torch.max(out, 1)  # calculate prediction,maximum 1
        acc += (pred == label).float().mean()  # calculate accuracy
    print(
        f'Test Loss: {loss_total / len(test_loader):.6f}, Acc: {acc / len(test_loader):.6f}\n')  # print process
    test_loss.append(loss_total / len(test_loader))  # record loss each epoch
    test_acc.append(float(acc / len(test_loader)))  # record accuracy each epoch
    return


for epoch in range(1, epochs + 1):  # for every epoch
    print("Epoch:", epoch)  # print epoch index
    train(0)  # call out training
    test(0)  # call out testing

# plot
sns.set()
fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(211)
# convert list into df
df1 = pd.DataFrame({'train_loss': train_loss, 'test_loss': test_loss})
ax1 = sns.lineplot(data=df1)
ax1.set_xlabel("epoches")
ax1.set_ylabel("loss")
plt.title("Loss Change on BDNN Model")
ax2 = fig.add_subplot(212)
df2 = pd.DataFrame({'train_acc': train_acc, 'test_acc': test_acc})
ax2 = sns.lineplot(data=df2)
plt.title("Accuracy Change on BDNN Model")
ax2.set_xlabel("epoches")
ax2.set_ylabel("accuracy")
plt.show()
