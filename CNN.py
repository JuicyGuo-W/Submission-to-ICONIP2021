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

# reshape train and test sets to 2D inputs
X_train = X_train.reshape((25, 13, 14))
X_test = X_test.reshape((25, 13, 14))

# DataLoader
train = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train, batch_size=5, shuffle=True)
test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size=5, shuffle=True)

# record history to plot
train_loss = []  # training loss
test_loss = []  # testing loss
train_acc = []  # training accuracy
test_acc = []  # testing accuracy


# cnn Net
class cnn_net(nn.Module):
    def __init__(self):  # define a cnn model
        super(cnn_net, self).__init__()  # inherit nn.Module
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)  # convolutional layer 1, channels 1->5, kernel size 3
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)  # convolutional layer 2, channels 5->10, kernel size 5
        self.mp = nn.MaxPool2d(2, padding=1)  # maxpooling 1, 2*2 maximum, padding 1
        self.relu = nn.ReLU()  # relu
        self.fc = nn.Linear(40, 5)  # fully-connected layer, inputs 40 -> outputs 5
        self.softmax = nn.LogSoftmax()  # softmax classifier

    def forward(self, x):  # forward process
        in_size = x.size(0)  # record input zie
        x = self.relu(self.mp(self.conv1(x)))  # conv1->maxpooling->relu
        x = self.relu(self.mp(self.conv2(x)))  # conv2->maxpooling->relu
        x = x.view(in_size, -1)  # flatten inputs
        x = self.fc(x)  # feed into fully-connected layer
        x = self.softmax(x)  # softmax classifier
        return x  # return outputs


# built a model and set an optimizer
model = cnn_net()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training
def train(flag):  # training method
    model.train()  # set train mode
    loss_total = 0.0  # record loss for each epoch
    acc = 0.00  # record accuracy for each epoch
    for index, train_data in enumerate(train_loader, 1):  # for each batch
        optimizer.zero_grad()  # set zero gradients
        features, label = train_data  # get features and labels
        out = model(features[:, None, ...])  # sequence the input dimension,and calculate outputs
        loss = F.cross_entropy(out, label)  # calculate loss in each batch
        loss_total += loss.item()  # calculate loss in each epoch
        _, pred = torch.max(out, 1)  # calculate prediction,maximum 1
        acc += (pred == label).float().mean()  # calculate accuracy
        loss.backward()  # back propagation
        optimizer.step()  # update weights
    print(f'Train Loss: {loss_total / index:.6f}, Acc: {acc / index:.6f}')  # print process
    train_loss.append(loss_total / index)  # record loss each epoch
    train_acc.append(float(acc / index))  # record accuracy each epoch
    return


def test(flag):  # testing method
    model.eval()  # test mode
    loss_total = 0.0  # record loss
    acc = 0.0  # record accuracy
    for test in test_loader:  # get data from test_loader
        features, label = test  # get features and label from test loader
        with torch.no_grad():
            out = model(features[:, None, ...])  # sequence the input dimension,and calculate outputs
            loss = F.cross_entropy(out, label)  # calculate loss for each batch
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
plt.title("Loss Change on CNN Model")
ax2 = fig.add_subplot(212)
df2 = pd.DataFrame({'train_acc': train_acc, 'test_acc': test_acc})
ax2 = sns.lineplot(data=df2)
plt.title("Accuracy Change on CNN Model")
ax2.set_xlabel("epoches")
ax2.set_ylabel("accuracy")
plt.show()
