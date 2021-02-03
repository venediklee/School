import numpy as np
from math import log, inf#for log function
def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that paragraph.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    vocab = set()
    for paragraph in data:
        for word in paragraph:
            vocab.add(word)
    return vocab

def train(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that paragraph.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta, pi. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all of the words in vocab and the values are their estimated probabilities.
             pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    #find unique class names from train labels
    classNames = set(train_labels)
    #find word count per class(including duplicates)
    wordCountsPerClass = dict.fromkeys(classNames,0)
    for paragraphID,paragraph in enumerate(train_data):
        wordCountsPerClass[train_labels[paragraphID]]+=len(paragraph)
    #find unique word count
    totalUniqueWordCount = len(vocab)

    theta = dict.fromkeys(classNames,0)#initialize theta and sub dictionaries
    for className in classNames:
        #initialize words with 1(normalized with multiplier), since smoothing constant is 1
        theta[className] = dict.fromkeys(vocab,GetMultiplier(className,totalUniqueWordCount,wordCountsPerClass))

    for paragraphID,paragraph in enumerate(train_data):
        #calculate 1/denominator, i.e multiplier for this label
        className = train_labels[paragraphID]
        multiplier = GetMultiplier(className, totalUniqueWordCount, wordCountsPerClass)
        for word in paragraph:
            theta[className][word]+=multiplier
    pi = dict.fromkeys(classNames,0)
    for className in classNames:
        pi[className] = train_labels.count(className) / len(train_labels)
    return theta, pi

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that paragraph.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    scores = [[] for _ in range(len(test_data))]
    for paragraphID,paragraph in enumerate(test_data):
        scores[paragraphID] = [("className",0) for _ in range(len(theta))]
        for classID,className in enumerate(theta.keys()):
            score = log(pi[className])
            for word in paragraph:
                if(word in theta[className]):
                    score+=log(theta[className][word])
            scores[paragraphID][classID] = (score,className)
    return scores

def GetMultiplier(className, totalUniqueWordCount, wordCountsPerClass):
    return 1.0 / (wordCountsPerClass[className] + totalUniqueWordCount)

def GetData(dataFile):
    data = dataFile.read().translate({ord(badChar):None for badChar in ".,():;[]'!*"}).splitlines()#remove each bad character, split with newline Then split with " "
    for i in range(len(data)):
        data[i] = data[i].split()#splits on space
    return data

def GetLabels(labelFile):
    labels = labelFile.readlines()
    return [label.strip() for label in labels]


def main():
    trainDataFile = open('hw4_data/news/train_data.txt','r')
    trainLabelFile = open('hw4_data/news/train_labels.txt','r')
    testLabelFile = open('hw4_data/news/test_labels.txt','r')
    testDataFile = open('hw4_data/news/test_data.txt','r')

    trainData = GetData(trainDataFile)
    trainLabels = GetLabels(trainLabelFile)
    testData = GetData(testDataFile)
    testLabels = GetLabels(testLabelFile)

    trainDataFile.close()
    trainLabelFile.close()
    testLabelFile.close()
    testDataFile.close()

    trainVocabulary = vocabulary(trainData)
    theta,pi = train(trainData,trainLabels,trainVocabulary)

    scores = test(theta,pi,trainVocabulary,testData)

    correctGuessCount = 0
    for paragraphID,score in enumerate(scores):
        highestGuessName = ""
        highestGuessValue = -inf
        for classProbability,className in score:
            if(classProbability > highestGuessValue):
                highestGuessName = className
                highestGuessValue = classProbability
        if(highestGuessName == testLabels[paragraphID]):
            correctGuessCount+=1
    print(correctGuessCount / len(testLabels))

if __name__ == '__main__':
    main()
    exit