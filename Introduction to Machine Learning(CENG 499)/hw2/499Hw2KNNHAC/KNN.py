import numpy as np
import matplotlib.pyplot as plt

def PlotKnnAccuracies(train_data, train_labels):
    KValues = np.arange(1,200,2)#1,3,5..199
    foldCount = 10
    foldDataCount = int(train_data.shape[0] / foldCount)#data count per fold
    
    correctGuessCounts = [0] * len(KValues)#error count per KValue
    
    for foldIndex in range(foldCount):#for each fold
        foldData = train_data[foldDataCount * foldIndex:foldDataCount * (foldIndex + 1)]# 1/foldCount of the data
        foldLabels = train_labels[foldDataCount * foldIndex:foldDataCount * (foldIndex + 1)]# 1/foldCount of the labels
        
        mask = np.ones(len(train_data),dtype=bool)#create a mask to filter out the foldData from the train_data
        mask[foldDataCount * foldIndex:foldDataCount * (foldIndex + 1)] = False
        pinnedData = train_data[mask]# rest of the data
        pinnedLabels = train_labels[mask]# rest of the labels
    
        for KValueIndex in range(len(KValues)):
            KValue = KValues[KValueIndex]
            correctGuessCount = 0
            for i in range(foldData.shape[0]):#foreach data in fold
                smallestDistances = [float("inf")] * KValue
                smallestDistanceLabels = [-1] * KValue
                for j in range(pinnedData.shape[0]):#find the minimum distances
                    distance = np.linalg.norm(foldData[i] - pinnedData[j])
                    for smallDistanceIndex in range(len(smallestDistances)):
                        if(distance < smallestDistances[smallDistanceIndex]):
                            smallestDistances[smallDistanceIndex] = distance
                            smallestDistanceLabels[smallDistanceIndex] = pinnedLabels[j]
                            break
                predictedLabel = max(smallestDistanceLabels,key=smallestDistanceLabels.count)
                if(predictedLabel == foldLabels[i]):
                    correctGuessCount +=1
            correctGuessCounts[KValueIndex]+=correctGuessCount
            print("correct guess count for K=" + str(KValue) + " at foldIndex=" + str(foldIndex) + " = " + str(correctGuessCount))
    
    #convert to percent accuracy
    for i in range(len(correctGuessCounts)):
        correctGuessCounts[i] = 100 * correctGuessCounts[i] / ((foldCount) * foldDataCount)
    plt.xlabel("KValues")
    plt.ylabel("average accuracy %")
    plt.axis([0,199,0,100])
    plt.plot(KValues,correctGuessCounts,label="accuracy per KValue", marker='o',fillstyle="none")
    plt.legend()
    plt.show()

def TestAccuracy(train_data,train_labels, test_data,test_labels):
    
    correctGuessCount = 0
    
    
    KValue = 3
    for i in range(test_data.shape[0]):#foreach data in test_data
        smallestDistances = [float("inf")] * KValue
        smallestDistanceLabels = [-1] * KValue
        for j in range(train_data.shape[0]):#find the minimum distances
            distance = np.linalg.norm(test_data[i] - train_data[j])
            for smallDistanceIndex in range(len(smallestDistances)):
                if(distance < smallestDistances[smallDistanceIndex]):
                    smallestDistances[smallDistanceIndex] = distance
                    smallestDistanceLabels[smallDistanceIndex] = train_labels[j]
                    break
        predictedLabel = max(smallestDistanceLabels,key=smallestDistanceLabels.count)
        if(predictedLabel == test_labels[i]):
            correctGuessCount +=1
    
    #convert to percent accuracy
    correctGuessCount = 100 * correctGuessCount / (test_data.shape[0])
    print("accuracy on test set=" + str(correctGuessCount) + "%")

def main():
    train_data = np.load('hw2_data/knn/train_data.npy')
    train_labels = np.load('hw2_data/knn/train_labels.npy')
    test_data = np.load('hw2_data/knn/test_data.npy')
    test_labels = np.load('hw2_data/knn/test_labels.npy')

    PlotKnnAccuracies(train_data, train_labels)#best KValue=3
    TestAccuracy(train_data,train_labels,test_data,test_labels)


if __name__ == '__main__':
    main()
    exit