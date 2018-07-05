import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.functional as F
from roc import calculate_roc

class Module(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Module, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, 10),
                                    nn.ReLU(),
                                    nn.Linear(10, num_classes))

    def forward(self, x):
        out = self.linear(x)
        return out


def data_loader(data, target, batch_size=20, shuffle=True):
    data_tensor = torch.FloatTensor(data)
    target_tensor = torch.LongTensor(target)
    train = data_utils.TensorDataset(data_tensor, target_tensor)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def train(model, train_loader, learning_rate=0.001, num_epochs=100, debug=False):
    # model = LogisticRegression(input_size=input_size, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (datas, labels) in enumerate(train_loader):
            datas = Variable(datas)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(datas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if debug:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (
                    epoch + 1, num_epochs, i + 1, len(train_loader.dataset) // train_loader.batch_size + 1,
                    loss.data[0]))

    return model


def predict(x):
    pass


def test(model, test_loader):
    correct = 0
    total = 0
    scores = torch.FloatTensor([])
    for datas, labels in test_loader:
        datas = Variable(datas)
        outputs = model(datas)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        outputs = F.softmax(outputs, dim=1)
        scores = torch.cat((scores, outputs.data[:, 1]))
    print('Accuracy of the model: %d %% ' % (100 * correct / total))
    return scores.numpy()


def train_and_test(module, train_loader, test_loader):
    train(module, train_loader)
    scores = test(module, test_loader)
    calculate_roc(test_loader.dataset.target_tensor.numpy(), scores)
