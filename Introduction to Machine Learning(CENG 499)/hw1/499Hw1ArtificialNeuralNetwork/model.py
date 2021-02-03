import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import MyDataset

class TestModel(nn.Module):# F.relu torch.sigmoid torch.tanh
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64,256)
        self.fc2 = nn.Linear(256,100)
        self.fc3 = nn.Linear(1024,100)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        x = torch.tanh(x)
        x = self.fc2(x)

        x = torch.log_softmax(x,dim=1)
        return x


class OneLayerModel(nn.Module):
    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64,100)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        x = torch.log_softmax(x,dim=1)
        return x


class TwoLayerModel(nn.Module):
    def __init__(self, hiddenLayerSize=256, hiddenLayerActivationFunction=torch.sigmoid):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64,hiddenLayerSize)
        self.fc2 = nn.Linear(hiddenLayerSize,100)
        self.hiddenLayerSize = hiddenLayerSize
        self.hiddenLayerActivationFunction = hiddenLayerActivationFunction

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        x = self.hiddenLayerActivationFunction(x)
        x = self.fc2(x)

        x = torch.log_softmax(x,dim=1)
        return x

class ThreeLayerModel(nn.Module):
    def __init__(self, hiddenLayerSize=256, hiddenLayerActivationFunction=torch.sigmoid):
        super(ThreeLayerModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64,hiddenLayerSize)
        self.fc2 = nn.Linear(hiddenLayerSize,hiddenLayerSize)
        self.fc3 = nn.Linear(hiddenLayerSize,100)
        self.hiddenLayerSize = hiddenLayerSize
        self.hiddenLayerActivationFunction = hiddenLayerActivationFunction

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        x = self.hiddenLayerActivationFunction(x)
        x = self.fc2(x)
        
        x = self.hiddenLayerActivationFunction(x)
        x = self.fc3(x)

        x = torch.log_softmax(x,dim=1)
        return x

if __name__ == '__main__':
    torch.set_printoptions(threshold=5000)
    transforms = T.Compose([T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),])
    dataset = MyDataset('data','train',transforms)
    dataLoader = DataLoader(dataset,batch_size=64,shuffle=True,num_workers=4,drop_last=True)#drop last since 10k can't be divided by 64
    model = TestModel()
    for images, labels in dataLoader:
        prediction = model(images)
        print(prediction)
        exit()