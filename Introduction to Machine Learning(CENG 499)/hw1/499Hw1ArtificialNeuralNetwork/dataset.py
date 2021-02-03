import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class TestDataset(Dataset):
    def __init__(self, datasetPath,split, transforms):
        imagePaths = 'C:\\Users\\Bright Lord\\Desktop\\499\\hw1\\499Hw1ArtificialNeuralNetwork\\data\\' + split#os.path.join(datasetPath,split)
        self.data = []
        with open(os.path.join(imagePaths,'labels.txt'),'r') as file:
            for line in file:
                imageName = line.rstrip('\n')
                imagePath = os.path.join(imagePaths, imageName)
                self.data.append(imagePath)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imagePath = self.data[index]
        image = Image.open(imagePath)
        image = self.transforms(image)
        return image

class MyDataset(Dataset):
    def __init__(self, datasetPath,split, transforms):
        imagePaths = 'C:\\Users\\Bright Lord\\Desktop\\499\\hw1\\499Hw1ArtificialNeuralNetwork\\data\\' + split#os.path.join(datasetPath,split)
        self.data = []
        with open(os.path.join(imagePaths,'labels.txt'),'r') as file:
            for line in file:
                imageName, label = line.split()
                imagePath = os.path.join(imagePaths, imageName)
                label = int(label)
                self.data.append((imagePath,label))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imagePath = self.data[index][0]
        label = self.data[index][1]
        image = Image.open(imagePath)
        image = self.transforms(image)
        return image, label


if __name__ == '__main__':
    transforms = T.Compose([T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),])
    dataset = MyDataset('data','train',transforms)
    dataLoader = DataLoader(dataset,batch_size=64,shuffle=True,num_workers=4,drop_last=True)#drop last since 10k can't be divided by 64
    for images, labels in dataLoader:
        print(images.size())
        print(labels)
        exit(0)