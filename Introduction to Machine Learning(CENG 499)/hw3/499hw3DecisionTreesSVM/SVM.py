from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from draw import draw_svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,plot_confusion_matrix
import matplotlib.pyplot as plt


def Task1():
    train_data = np.load('hw3_data/linsep/train_data.npy')
    train_labels = np.load('hw3_data/linsep/train_labels.npy')
    cValues = [0.01,0.1,1,10,100]
    for cValue in cValues:
        clf = SVC(kernel='linear', gamma='auto',C=cValue)
        clf.fit(train_data, train_labels)
        draw_svm(clf,train_data,train_labels,
                 min(train_data[:,0]) - 1,max(train_data[:,0]) + 1,
                 min(train_data[:,1]) - 1,max(train_data[:,1]) + 1,
                 "SVMOutputs/Task1/linear kernel with C=" + str(cValue) + ".png")

def Task2():
    train_data = np.load('hw3_data/nonlinsep/train_data.npy')
    train_labels = np.load('hw3_data/nonlinsep/train_labels.npy')

    kernels = ['linear', 'rbf','poly','sigmoid']
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='auto')
        clf.fit(train_data, train_labels)
        draw_svm(clf,train_data,train_labels,
                 min(train_data[:,0]) - 1,max(train_data[:,0]) + 1,
                 min(train_data[:,1]) - 1,max(train_data[:,1]) + 1,
                 "SVMOutputs/Task2/kernel =" + kernel + ".png")

def Task3():
    train_data = np.load('hw3_data/catdog/train_data.npy')
    train_labels = np.load('hw3_data/catdog/train_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1

    cValues = [0.01,0.1,1,10,100]
    predictionAccuracies = []
    for cValue in cValues:
        clf = SVC(kernel='linear',C=cValue)
        testScores = cross_validate(clf,train_data,train_labels,cv=5)['test_score']
        predictionAccuracies.append(sum(testScores) / len(testScores))
    PrintPredictionAccuracies(predictionAccuracies,'linear','-')
            
    gammas = [0.00001,0.0001,0.001,0.01,0.1,1]
    kernels = ['rbf','poly','sigmoid']
    for kernel in kernels:
        for gamma in gammas:
            predictionAccuracies = []
            for cValue in cValues:
                clf = SVC(kernel=kernel,C=cValue,gamma=gamma)
                testScores = cross_validate(clf,train_data,train_labels,cv=5)['test_score']
                predictionAccuracies.append(sum(testScores) / len(testScores))
            PrintPredictionAccuracies(predictionAccuracies,kernel,gamma)

def Task3AccuracyForTestData():
    train_data = np.load('hw3_data/catdog/train_data.npy')
    train_labels = np.load('hw3_data/catdog/train_labels.npy')
    test_data = np.load('hw3_data/catdog/test_data.npy')
    test_labels = np.load('hw3_data/catdog/test_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1
    inverseHalfMax = 1.0 / (test_data.max() / 2.0)
    test_data = test_data * inverseHalfMax - 1#test_data=test_data/(max/2) - 1

    #train the clf with best accuracy on train data and test it with test data
    kernel = 'poly'
    gamma = 0.01
    cValue = 0.01
    clf = SVC(kernel=kernel,C=cValue,gamma=gamma)
    clf.fit(train_data,train_labels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize=True)
    print("accuracy with " + kernel + " kernel at " + str(gamma) + " gamma and " + str(cValue) + " cValue for test data = " + str(accuracy))

    gamma = 0.001
    cValue = 10
    clf = SVC(kernel=kernel,C=cValue,gamma=gamma)
    clf.fit(train_data,train_labels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize=True)
    print("accuracy with " + kernel + " kernel at " + str(gamma) + " gamma and " + str(cValue) + " cValue for test data = " + str(accuracy))

def PrintPredictionAccuracies(predictionAccuracies,kernel,gamma):
    print("For " + kernel + " kernel and gamma " + str(gamma) + " average accuracies at 5 fold cross validation per c value=", *predictionAccuracies)

def Task4WithoutOversampling():
    train_data = np.load('hw3_data/catdogimba/train_data.npy')
    train_labels = np.load('hw3_data/catdogimba/train_labels.npy')
    test_data = np.load('hw3_data/catdogimba/test_data.npy')
    test_labels = np.load('hw3_data/catdogimba/test_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1
    inverseHalfMax = 1.0 / (test_data.max() / 2.0)
    test_data = test_data * inverseHalfMax - 1#test_data=test_data/(max/2) - 1

    clf = SVC(kernel='rbf',C=1)
    clf.fit(train_data,train_labels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize=True)
    print("accuracy with rbf kernel at 1 cValue for test data = " + str(accuracy))

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(clf,test_data, test_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title("confusion matrix of test data with accuracy=" + str(accuracy))

    print("confusion matrix of test data")
    print(disp.confusion_matrix)
    plt.savefig("SVMOutputs/Task4/confusion matrix of test data.png")

def Task4WithOversampling():
    train_data = np.load('hw3_data/catdogimba/train_data.npy')
    train_labels = np.load('hw3_data/catdogimba/train_labels.npy')
    test_data = np.load('hw3_data/catdogimba/test_data.npy')
    test_labels = np.load('hw3_data/catdogimba/test_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1
    inverseHalfMax = 1.0 / (test_data.max() / 2.0)
    test_data = test_data * inverseHalfMax - 1#test_data=test_data/(max/2) - 1

    uniqueValues,counts = np.unique(train_labels,return_counts=True)
    minCount = counts[0]
    minCountIndex = 0
    maxCount = counts[0]
    maxCountIndex = 0
    for i in range(len(counts)):
        if(counts[i] < minCount):
            minCount = counts[i]
            minCountIndex = i
        if(counts[i] > maxCount):
            maxCount = counts[i]
            maxCountIndex = i
    countDifference = maxCount - minCount
    minorityIndices = train_labels == uniqueValues[minCountIndex]
    appendCount = int(countDifference / sum(minorityIndices))

    minorityData = np.tile(train_data[minorityIndices],(appendCount,1))
    newData = np.concatenate((train_data,minorityData),axis=0)
    minorityLabels = np.tile(uniqueValues[minCountIndex],(appendCount * sum(minorityIndices)))
    newLabels = np.concatenate((train_labels,minorityLabels),axis=0)

    clf = SVC(kernel = 'rbf',C = 1)
    clf.fit(newData,newLabels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize = True)
    print("(oversampled) accuracy with rbf kernel at 1 cValue for test data = " + str(accuracy))

    np.set_printoptions(precision = 2)
    disp = plot_confusion_matrix(clf,test_data, test_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title("(oversampled) confusion matrix of test data with accuracy=" + str(accuracy))

    print("(oversampled) confusion matrix of test data")
    print(disp.confusion_matrix)
    plt.savefig("SVMOutputs/Task4/(oversampled) confusion matrix of test data.png")

def Task4WithUndersampling():
    train_data = np.load('hw3_data/catdogimba/train_data.npy')
    train_labels = np.load('hw3_data/catdogimba/train_labels.npy')
    test_data = np.load('hw3_data/catdogimba/test_data.npy')
    test_labels = np.load('hw3_data/catdogimba/test_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1
    inverseHalfMax = 1.0 / (test_data.max() / 2.0)
    test_data = test_data * inverseHalfMax - 1#test_data=test_data/(max/2) - 1

    uniqueValues,counts = np.unique(train_labels,return_counts=True)
    minCount = counts[0]
    minCountIndex = 0
    maxCount = counts[0]
    maxCountIndex = 0
    for i in range(len(counts)):
        if(counts[i] < minCount):
            minCount = counts[i]
            minCountIndex = i
        if(counts[i] > maxCount):
            maxCount = counts[i]
            maxCountIndex = i
    countDifference = maxCount - minCount
    minorityIndices = train_labels == uniqueValues[minCountIndex]
    majorityIndices = train_labels == uniqueValues[maxCountIndex]

    minorityData = train_data[minorityIndices]
    majorityData = train_data[majorityIndices][:len(minorityData)]

    newData = np.concatenate((minorityData,majorityData),axis=0)
    minorityLabels = np.tile(uniqueValues[minCountIndex],len(minorityData))
    majorityLabels = np.tile(uniqueValues[maxCountIndex],len(majorityData))
    newLabels = np.concatenate((minorityLabels,majorityLabels),axis = 0)

    clf = SVC(kernel = 'rbf',C = 1)
    clf.fit(newData,newLabels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize = True)
    print("(undersampled) accuracy with rbf kernel at 1 cValue for test data = " + str(accuracy))

    np.set_printoptions(precision = 2)
    disp = plot_confusion_matrix(clf,test_data, test_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title("(undersampled) confusion matrix of test data with accuracy=" + str(accuracy))

    print("(undersampled) confusion matrix of test data")
    print(disp.confusion_matrix)
    plt.savefig("SVMOutputs/Task4/(undersampled) confusion matrix of test data.png")

def Task4WithClassWeight():
    train_data = np.load('hw3_data/catdogimba/train_data.npy')
    train_labels = np.load('hw3_data/catdogimba/train_labels.npy')
    test_data = np.load('hw3_data/catdogimba/test_data.npy')
    test_labels = np.load('hw3_data/catdogimba/test_labels.npy')

    #normalize
    inverseHalfMax = 1.0 / (train_data.max() / 2.0)
    train_data = train_data * inverseHalfMax - 1#train_data=train_data/(max/2) - 1
    inverseHalfMax = 1.0 / (test_data.max() / 2.0)
    test_data = test_data * inverseHalfMax - 1#test_data=test_data/(max/2) - 1

    clf = SVC(kernel='rbf',C=1,class_weight='balanced')
    clf.fit(train_data,train_labels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels,predictions,normalize=True)
    print("(classweighted) accuracy with rbf kernel at 1 cValue for test data = " + str(accuracy))

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(clf,test_data, test_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title("(classweighted) confusion matrix of test data with accuracy=" + str(accuracy))

    print("(classweighted) confusion matrix of test data")
    print(disp.confusion_matrix)
    plt.savefig("SVMOutputs/Task4/(classweighted) confusion matrix of test data.png")

def main():
    Task1()
    Task2()
    Task3()
    Task3AccuracyForTestData()
    Task4WithoutOversampling()
    Task4WithOversampling()
    Task4WithUndersampling()
    Task4WithClassWeight()

if __name__ == '__main__':
    main()
    exit