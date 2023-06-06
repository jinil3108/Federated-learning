import torch
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights

import dla
import mobilenetv2
import util
import torch.optim as optimize
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import resnet_implementer
import vgg
from torchsummary import summary
from matplotlib import pyplot as plt
torch.backends.cudnn.benchmark=True


def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


# Server Aggregation Function
def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    # This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc


# Using it for image transformation of training model.
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Using it for image transformation of testing model.
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Data Loading Takes Place with the help of data loader for every client in the array.
dataloader = []

for i in range(util.num_clients):
    dataset = torchvision.datasets.ImageFolder("Dataset/train/"+str(i+1), transform=train_transform)
    dataloader.append(torch.utils.data.DataLoader(dataset, batch_size=util.batch_size, shuffle=True))


# Loading into the DataLoader for testing.
test_set = torchvision.datasets.ImageFolder("Dataset/test", transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=util.batch_size, shuffle=True)


user = int(input("Enter your choice\n1. Resnet-50\n2. VGG-19\n3. MobileNetV2\n4. DLA\n"))

# Creating the global model and client model.
if user == 1:
    global_model = resnet_implementer.ResNet50().cuda()
    client_models = [resnet_implementer.ResNet50().cuda() for _ in range(util.num_selected)]
elif user == 2:
    global_model = vgg.VGG('VGG19').cuda()
    client_models = [vgg.VGG('VGG19').cuda() for _ in range(util.num_selected)]
elif user == 3:
    global_model = mobilenetv2.MobileNetV2().cuda()
    client_models = [mobilenetv2.MobileNetV2().cuda() for _ in range(util.num_selected)]
elif user == 4:
    global_model = dla.DLA().cuda()
    client_models = [dla.DLA().cuda() for _ in range(util.num_selected)]

for model in client_models:
    model.load_state_dict(global_model.state_dict())


# Optimizers
opt = [optimize.SGD(model.parameters(), lr=0.1) for model in client_models]


# List containing info about learning #########
losses_train = []
losses_test = []
acc_test = []


# Running Federated Learning.
for r in range(util.num_rounds):
    # select random clients
    client_idx = np.random.permutation(util.num_clients)[:util.num_selected]
    # client update
    loss = 0
    for i in tqdm(range(util.num_selected)):
        loss += client_update(client_models[i], opt[i], dataloader[client_idx[i]], epoch=util.epochs)

    losses_train.append(loss)

    # server aggregate.
    server_aggregate(global_model, client_models)

    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / util.num_selected, test_loss, acc))

print("Best Accuracy: ", np.max(acc_test)*100, "%")