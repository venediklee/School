import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import MyDataset, TestDataset
from model import OneLayerModel,TwoLayerModel,ThreeLayerModel,TestModel
import matplotlib.pyplot as plt
import numpy as np
import copy

def drawGraph(model, optimizer,epochs, transforms, device,useCuda):
    minLoss = float('inf')
    evaluationLoss = 0
    correctGuesses = 0

    trainDataset = MyDataset('data','train',transforms)
    trainDataset.data = trainDataset.data[:int(len(trainDataset.data) * 0.8)]#gets the first 80%
    trainDataloader = DataLoader(trainDataset,batch_size = 64,shuffle = True,num_workers = 0,drop_last = True,#drop last since 10k can't be divided by 64
                                 pin_memory=True if useCuda else False)

    validationDataset = MyDataset('data','train',transforms)
    validationDataset.data = validationDataset.data[int(len(validationDataset.data) * 0.8) :]#gets the last 20%
    validationDataLoader = DataLoader(validationDataset,batch_size = 64,shuffle = True,num_workers = 0,drop_last = True,#drop last since 10k can't be divided by 64
                                 pin_memory = True if useCuda else False)

    trainLossValues = []
    validationLossValues = []
    for epochID in range(epochs):
        totalLoss = 0
        batchCount = 0
        for images, labels in trainDataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = model(images)
            loss = F.nll_loss(prediction,labels)
            loss.backward()
            optimizer.step()
            totalLoss+=loss.item()
            batchCount+=1
        averageLossAtEpoch = totalLoss / batchCount
        trainLossValues.append(averageLossAtEpoch)


        model.eval()
        with torch.no_grad():
            evaluationLoss = 0
            evaluationBatchCount = 0
            for images, labels in validationDataLoader:
                images = images.to(device)
                labels = labels.to(device)
                prediction = model(images)
                loss = F.nll_loss(prediction,labels)
                evaluationLoss+=loss.item()
                predictedLabels = prediction.data.max(1,keepdim=True)[1]#gets the class with the highest prediction chance
                correctGuesses += predictedLabels.eq(labels).sum()#sums the count of correct guesses
                evaluationBatchCount+=1
        evaluationLoss/=evaluationBatchCount
        validationLossValues.append(evaluationLoss)

    plt.plot(np.array(list(range(1, epochs + 1))),np.array(trainLossValues),label="training loss")
    plt.plot(np.array(list(range(1,epochs + 1))),np.array(validationLossValues),label = "validation loss")
    plt.xlabel("epochs")
    plt.ylabel("values")
    plt.legend()
    plt.show()

def printPredictionLabels(model,optimizer,dataLoader,device):
    model.eval()
    labels = []
    with torch.no_grad():
        for images in dataLoader:
            images = images.to(device)
            prediction = model(images)
            predictedLabels = prediction.data.max(1,keepdim=True)[1]#gets the class with the highest prediction chance
            labels+=predictedLabels.squeeze().tolist()
    labels = list(map(str,labels))
    with open(os.path.join('C:\\Users\\Bright Lord\\Desktop\\499\\hw1\\499Hw1ArtificialNeuralNetwork\\data\\test','predictions.txt'),'w') as file:
        for i in range(len(labels)):
            file.write(str(i) + ".png " + labels[i] + "\n")

def evaluate(model, optimizer, dataLoader, device):
    model.eval()
    evaluationLoss = 0
    correctGuesses = 0
    batchCount = 0
    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            loss = F.nll_loss(prediction,labels)
            evaluationLoss+=loss.item()
            predictedLabels = prediction.data.max(1,keepdim=True)[1]#gets the class with the highest prediction chance
            # sums the count of correct guesses
            predictedLabels = predictedLabels.squeeze()
            for i in range(len(labels)):
                if(predictedLabels[i] == labels[i]):
                    correctGuesses += 1
            batchCount+=1
    evaluationLoss/=batchCount
    print("average evaluation loss = {}, with {} correct guesses. Accuracy = {}%".
          format(evaluationLoss,correctGuesses,100 * correctGuesses / len(dataLoader.dataset)))
  
def save(model, optimizer, dataLoader, epochs, device, fileName):
    model.train()
    minLoss = float('inf')
    for epochID in range(epochs):
        totalLoss = 0
        batchCount = 0
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = model(images)
            loss = F.nll_loss(prediction,labels)
            loss.backward()
            optimizer.step()
            totalLoss+=loss.item()
            batchCount+=1
        averageLossAtEpoch = totalLoss / batchCount
        if(averageLossAtEpoch < minLoss):
            minLoss = min(minLoss,averageLossAtEpoch)
    torch.save(model.state_dict(), fileName)
    return minLoss

def train(model, optimizer, dataLoader, epochs, device):
    model.train()
    minLoss = float('inf')
    for epochID in range(epochs):
        totalLoss = 0
        batchCount = 0
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = model(images)
            loss = F.nll_loss(prediction,labels)
            loss.backward()
            optimizer.step()
            totalLoss+=loss.item()
            batchCount+=1
        averageLossAtEpoch = totalLoss / batchCount
        minLoss = min(minLoss,averageLossAtEpoch)
    return minLoss

def main():
    epochs = 15
    useCuda = False
    torch.manual_seed(1234)

    device = torch.device('cuda' if useCuda else 'cpu')
    transforms = T.Compose([T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),])
    trainDataset = MyDataset('data','train',transforms)
    trainDataset.data = trainDataset.data[:int(len(trainDataset.data) * 0.8)]#gets the first 80%
    trainDataLoader = DataLoader(trainDataset,batch_size = 64,num_workers = 0,drop_last = True,#drop last since 10k can't be divided by 64
                                 pin_memory=True if useCuda else False)
    
    validationDataset = MyDataset('data','train',transforms)
    validationDataset.data = validationDataset.data[int(len(validationDataset.data) * 0.8) :]#gets the last 20%
    validationDataLoader = DataLoader(validationDataset,batch_size = 64,num_workers = 0,drop_last = True,#drop last since 10k can't be divided by 64
                                 pin_memory = True if useCuda else False)

    
    ##########used for saving the model
    #trainDataset.data.append(validationDataset.data)
    #model = ThreeLayerModel(1024,torch.tanh)
    #learningRate = 0.0003
    #optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    #loss =
    #save(model,optimizer,trainDataLoader,epochs,device,'lowestLossModelStateDict')
    #print("trained model with " + str(loss) + " loss")
    #exit()

    ###########used for evaluating the model
    #model = ThreeLayerModel(1024,torch.tanh)
    #model.load_state_dict(torch.load('lowestLossModelStateDict'))
    #learningRate = 0.0003
    #optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    #evaluate(model,optimizer,validationDataLoader, device)
    #exit()

    ##########used for drawing graph
    #model = ThreeLayerModel(1024,torch.tanh)
    #learningRate = 0.0003
    #optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    #drawGraph(model,optimizer,50,transforms,device,useCuda)
    #exit()

    ###########used for printing the test predictions
    #testDataset = TestDataset('data','test',transforms)
    #testDataLoader = DataLoader(testDataset,batch_size = 64,num_workers =
    #0,drop_last = False, pin_memory = True if useCuda else False)
    #model = ThreeLayerModel(1024,torch.tanh)
    #model.load_state_dict(torch.load('lowestLossModelStateDict'))
    #learningRate = 0.0003
    #optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    #printPredictionLabels(model,optimizer,testDataLoader, device)
    #exit()

    ###########used for calculating loss and accuracy without training
    ###########(for question 1.1 and 1.2)
    #model = ThreeLayerModel(1024,torch.tanh)
    #learningRate = 0.0003
    #optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    #evaluate(model,optimizer,validationDataLoader, device)
    #exit()



    #########used for finding the best hyperparameters

    models = [OneLayerModel(),#F.relu torch.sigmoid torch.tanh
            TwoLayerModel(256,F.relu),TwoLayerModel(256,torch.sigmoid),TwoLayerModel(256,torch.tanh),
            TwoLayerModel(512,F.relu),TwoLayerModel(512,torch.sigmoid),TwoLayerModel(512,torch.tanh),
            TwoLayerModel(1024,F.relu),TwoLayerModel(1024,torch.sigmoid),TwoLayerModel(1024,torch.tanh),
            ThreeLayerModel(256,F.relu),ThreeLayerModel(256,torch.sigmoid),ThreeLayerModel(256,torch.tanh),
            ThreeLayerModel(512,F.relu),ThreeLayerModel(512,torch.sigmoid),ThreeLayerModel(512,torch.tanh),
            ThreeLayerModel(1024,F.relu),ThreeLayerModel(1024,torch.sigmoid),ThreeLayerModel(1024,torch.tanh)]
    
    learningRates = [0.01, 0.003,0.001, 0.0003,0.0001, 0.00003]

    lossAndParameters = []

    for model in models:
        torch.save(model.state_dict(), 'preTrainingModelStateDict')#save model's zero state to reload it back.
        model = model.to(device)
        for learningRate in learningRates:
            model.load_state_dict(torch.load('preTrainingModelStateDict'))
            optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
            lossAndParameters.append((train(model,optimizer,trainDataLoader, epochs, device),model,learningRate))
            lossAndParameter = lossAndParameters[-1]
            print(("minLoss = {}, for {} with {} layer size with {} activation function at {} learning rate").
              format(lossAndParameter[0],type(lossAndParameter[1]),
                "-" if type(lossAndParameter[1]) == OneLayerModel else lossAndParameter[1].hiddenLayerSize,
                 "-" if type(lossAndParameter[1]) == OneLayerModel else lossAndParameter[1].hiddenLayerActivationFunction,
          lossAndParameter[2]))

    lossAndParameter = min(lossAndParameters,key = lambda t:t[0])#get the element with the lowest loss
    print(("training finished, minLoss = {}, for {} with {} layer size with {} activation function").
          format(lossAndParameter[0],type(lossAndParameter[1]),
                "-" if type(lossAndParameter[1]) == OneLayerModel else lossAndParameter[1].hiddenLayerSize,
                 "-" if type(lossAndParameter[1]) == OneLayerModel else lossAndParameter[1].hiddenLayerActivationFunction))

if __name__ == '__main__':
    main()
    exit()