import pickle
import math
import copy
from graphviz import Digraph

chiSquaredTableAt90Confidence = [2.706,4.605,6.251,7.779,9.236,10.645,12.017,13.362,14.684,15.987]#degree of freedom=1,2,...10
def divide(data, attr_index, attr_vals_list):
    """Divides the data into buckets according to the selected attr_index.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: A list that includes K data lists in it where K is the number
     of values that the attribute with attr_index can take
    """
    attributeValues = attr_vals_list[attr_index]
    buckets = [[] for _ in range(len(attributeValues))]
    for datum in data:
        for i in range(len(attributeValues)):
            if(datum[attr_index] == attributeValues[i]):
                buckets[i].append(datum)
                break
    return buckets


def entropy(data, attr_vals_list):
    """
    Calculates the entropy in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated entropy (float)
    """
    if(len(data) == 0):
        return 0
    #assume first datum's last attribute is positive, other value is negative
    lastIndex = len(attr_vals_list) - 1
    positiveValue = data[0][lastIndex]
    positiveCount = 0
    negativeCount = 0
    for datum in data:
        if(datum[lastIndex] == positiveValue):
            positiveCount +=1
        else:
            negativeCount+=1
    positiveProbability = float(positiveCount) / len(data)
    negativeProbability = float(negativeCount) / len(data)
    positiveEntropy = 0
    negativeEntropy = 0
    if(positiveProbability != 0):
        positiveEntropy = -positiveProbability * math.log(positiveProbability,2)
    if(negativeProbability != 0):
        negativeEntropy = -negativeProbability * math.log(negativeProbability,2)    
    return positiveEntropy + negativeEntropy

def info_gain(data, attr_index, attr_vals_list):
    """
    Calculates the information gain on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: information gain (float), buckets (the list returned from divide)
    """
    #info gain=entropy of the whole set - information gain of the selected attribute
    splits = divide(data,attr_index,attr_vals_list)
    averageEntropyPerSplit = [0 for _ in range(len(splits))]
    dataCount = len(data)
    for i in range(len(splits)):
        entropyOfSplit = entropy(splits[i],attr_vals_list)
        dataCountOfSplit = len(splits[i])
        averageEntropyPerSplit[i] = entropyOfSplit * float(dataCountOfSplit) / dataCount
    gain = entropy(data,attr_vals_list)
    for i in range(len(averageEntropyPerSplit)):
        gain-=averageEntropyPerSplit[i]
    return gain,splits


def gain_ratio(data, attr_index, attr_vals_list):
    """
    Calculates the gain ratio on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: gain_ratio (float), buckets (the list returned from divide)
    """
    #gain_ratio = info_gain/intrinsic info
    attributeValueCounts = [0 for _ in range(len(attr_vals_list[attr_index]))]
    splits = divide(data,attr_index,attr_vals_list)
    for i in range(len(splits)):
        attributeValueCounts[i] = len(splits[i])
    intrinsicInfo = 0
    for attributeValueCount in attributeValueCounts:
        if(attributeValueCount == 0):
            continue
        countRatio = float(attributeValueCount) / len(data)
        intrinsicInfo-=countRatio * math.log(countRatio,2)
    if(intrinsicInfo == 0):
        gainRatio = 0
    else:
        gainRatio = info_gain(data,attr_index,attr_vals_list)[0] / intrinsicInfo
    return gainRatio,splits


def gini(data, attr_vals_list):
    """
    Calculates the gini index in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated gini index (float)
    """
    giniValue = 1
    if(len(data) == 0):
        return giniValue
    #assume first datum's last attribute is positive, other value is negative
    lastIndex = len(attr_vals_list) - 1
    positiveValue = data[0][lastIndex]
    positiveCount = 0
    negativeCount = 0
    for datum in data:
        if(datum[lastIndex] == positiveValue):
            positiveCount +=1
        else:
            negativeCount+=1
    positiveProbability = float(positiveCount) / len(data)
    negativeProbability = float(negativeCount) / len(data)
    giniValue-=positiveProbability * positiveProbability
    giniValue-=negativeProbability * negativeProbability
    return giniValue


def avg_gini_index(data, attr_index, attr_vals_list):
    """
    Calculates the average gini index on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: average gini index (float), buckets (the list returned from divide)
    """
    if(len(data) == 0):
        return 0
    splits = divide(data,attr_index,attr_vals_list)
    averageGini = 0.0
    for i in range(len(splits)):
        averageGini+=len(splits[i]) * gini(splits[i],attr_vals_list)
    return averageGini / len(data),splits


def chi_squared_test(data, attr_index, attr_vals_list):
    """
    Calculated chi squared and degree of freedom between the selected attribute and the class attribute
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: chi squared value (float), degree of freedom (int)
    """
    #For each observed number in the table subtract the corresponding expected number (O — E).
    #Square the difference [ (O —E)**2 ].
    #Divide the squares obtained for each cell in the table by the expected number for that cell [ (O - E)**2 / E ].
    #Sum all the values for (O - E)**2 / E.  This is the chi square statistic.
    splits = divide(data,attr_index,attr_vals_list)
    observationTable = [[] for _ in range(len(splits))]#create table rows for each unique attribute value
    for splitID in range(len(splits)):
        observationTable[splitID] = [0 for _ in range(len(attr_vals_list[-1]))]#create table columns for each unique class attribute
        for datum in splits[splitID]:#fill table columns for each unique class attribute for selected splitID
            for classAttributeIndex in range(len(attr_vals_list[-1])):
                if(datum[-1] == attr_vals_list[-1][classAttributeIndex]):
                    observationTable[splitID][classAttributeIndex]+=1
                    break
    #remove zero columns and zero rows
    for iterationID in range(max(len(observationTable),len(observationTable[0]))):#used for restarting row/column removal iteration
        for rowID in range(len(observationTable)):
            isZeroRow = True
            for columnID in range(len(observationTable[0])):
                if(observationTable[rowID][columnID] > 0):
                    isZeroRow = False
                    break
            if(isZeroRow):
                observationTable.pop(rowID)
                break#restart iteration loop
        for columnID in range(len(observationTable[0])):
            isZeroColumn = True
            for rowID in range(len(observationTable)):
                if(observationTable[rowID][columnID] > 0):
                    isZeroColumn = False
                    break
            if(isZeroColumn):
                for rowID in range(len(observationTable)):
                    observationTable[rowID].pop(columnID)
                break#restart iteration loop
    #calculate expected values for each row-column tuple
    expectationTable = copy.deepcopy(observationTable)#easy way to create a new table
    for rowID in range(len(expectationTable)):
        rowSum = 0
        for datum in observationTable[rowID]:#find rowSum
            rowSum+=datum
        for columnID in range(len(observationTable[rowID])):
            columnSum = 0
            for rowID2 in range(len(expectationTable)):#find columnSum
                columnSum+=observationTable[rowID2][columnID]
            expectationTable[rowID][columnID] = float(rowSum * columnSum) / len(data)
    
    chiValue = 0
    for rowID in range(len(observationTable)):
        for columnID in range(len(observationTable[rowID])):
            observation = observationTable[rowID][columnID]
            expectation = expectationTable[rowID][columnID]
            chiValue+=((observation - expectation) ** 2) / expectation
    return chiValue,(len(observationTable) - 1) * (len(observationTable[0]) - 1) 

def GetHighestGainIndex(gains):
    highestInfoGainIndex = 0
    highestInfoGain = -1
    for i in range(len(gains)):
        if(gains[i] > highestInfoGain):
            highestInfoGain = gains[i]
            highestInfoGainIndex = i
    return highestInfoGainIndex

def GetLowestGainIndex(gains):
    lowestInfoGainIndex = 0
    lowestInfoGain = float('inf')
    for i in range(len(gains)):
        if(gains[i] < lowestInfoGain):
            lowestInfoGain = gains[i]
            lowestInfoGainIndex = i
    return lowestInfoGainIndex

def GetPositiveAndNegativeCount(attr_vals_list, data):
    positive = attr_vals_list[-1][0]
    positiveCount = 0
    negativeCount = 0
    for datum in data:
        if(datum[-1] == positive):
            positiveCount+=1
        else:
            negativeCount+=1
    return  positiveCount,negativeCount

def CreateTree(data, attr_vals_list, attr_names,metric="infoGain",prePruning=False):
    #tree= ["attributeName",output1,output2...,[edge1Name,edge2Name]] each output is the values the attribute can get.
    #   If the result is the same for each entry of a value then that output will be a leaf, containing the exact output.
    #       Otherwise it will be another tree
    #       The outputs are in the same order as attr_vals_list of that attribute
    if(metric == "infoGain"):
        gains = [info_gain(data,i,attr_vals_list)[0] for i in range(len(attr_vals_list) - 1)]
        highestGainIndex = GetHighestGainIndex(gains)
    elif(metric == "gainRatio"):
        gains = [gain_ratio(data,i,attr_vals_list)[0] for i in range(len(attr_vals_list) - 1)]
        highestGainIndex = GetHighestGainIndex(gains)
        if(prePruning):
            calculatedChi,degreeOfFreedom = chi_squared_test(data,highestGainIndex,attr_vals_list)
            if(calculatedChi < chiSquaredTableAt90Confidence[degreeOfFreedom - 1]):#data is not related, return leaf with the most repeated class variable
                positiveCount,negativeCount = GetPositiveAndNegativeCount(attr_vals_list, data)
                if(positiveCount > negativeCount):
                    return [attr_vals_list[-1][0] + "[" + str(positiveCount) + "," + str(negativeCount) + "]"]
                else:
                    return [attr_vals_list[-1][1] + "[" + str(positiveCount) + "," + str(negativeCount) + "]"]
    else:#metric==average gini index
        gains = [avg_gini_index(data,i,attr_vals_list)[0] for i in range(len(attr_vals_list) - 1)]
        highestGainIndex = GetLowestGainIndex(gains)#this is actually the lowest gain index
    
    tree = ['']
    tree[0] = attr_names[highestGainIndex]
    #split the current data into sub arrays
    splits = divide(data,highestGainIndex,attr_vals_list)
    for split in splits:
        if (len(split) == 0):#empty data, get max of parent
            positiveCount = 0
            negativeCount = 0
            for split2 in splits:
                p,n = GetPositiveAndNegativeCount(attr_vals_list, split2)
                positiveCount+=p
                negativeCount+=n
            if(positiveCount > negativeCount):
                tree.append(str(attr_vals_list[-1][0]) + "[" + str(positiveCount) + "," + str(negativeCount) + "]")
            else:
                tree.append(str(attr_vals_list[-1][1]) + "[" + str(positiveCount) + "," + str(negativeCount) + "]")
        elif(entropy(split,attr_vals_list) == 0):#leaf node
            positiveCount,negativeCount = GetPositiveAndNegativeCount(attr_vals_list, split)
            tree.append(split[0][-1] + "[" + str(positiveCount) + "," + str(negativeCount) + "]")
        else:
            tree.append(CreateTree(split,attr_vals_list,attr_names,metric,prePruning))
    #edges= [edge1,edge2...] follows each output's attribute value
    edges = []
    for attributeValue in attr_vals_list[highestGainIndex]:
        edges.append(attributeValue)
    tree.append(edges)
    return tree

def CreateTreeGraph(node, dot, indices, currentIndex):
    if(isinstance(node, str)):#if leaf
        return
    for i in range(1,len(node) - 1):#the childs of the tree
        childIndex = indices[-1] + 1
        indices.append(childIndex)
        childLabel = node[i]
        if(not isinstance(childLabel,str)):#if the label is not a string, get the first element of the array
            childLabel = node[i][0]
        dot.node(str(childIndex),childLabel)
        dot.edge(str(currentIndex),str(childIndex),node[-1][i - 1])
        CreateTreeGraph(node[i],dot,indices,childIndex)
        
def CreatePrePrunedTreeGraph(train_data,attr_vals_list, attr_names):
    tree = CreateTree(train_data,attr_vals_list, attr_names,"gainRatio",prePruning=True)
    dot = Digraph()
    dot.node('1',tree[0])#the root of the tree
    CreateTreeGraph(tree,dot, [1],1)
    try:
        dot.render(filename="DecisionTreeOutputs/gainRatioPrePruned",view=False)
    except Exception:
        pass
    return tree

def CreatePostPrunedTreeGraph(train_data,attr_vals_list, attr_names):
    
    trainSet = train_data[:int(len(train_data) * 0.8)]
    validationSet = train_data[int(len(train_data) * 0.8):]
    
    tree = CreateTree(trainSet,attr_vals_list, attr_names,"gainRatio",prePruning=False)

    totalPositive,totalNegative,prunedTree = PostPrune(tree,validationSet,attr_vals_list,attr_names)

    dot = Digraph()
    dot.node('1',prunedTree[0])#the root of the tree
    CreateTreeGraph(prunedTree,dot, [1],1)
    try:
        dot.render(filename="DecisionTreeOutputs/gainRatioPostPruned",view=False)
    except Exception:
        pass

    return tree

#returns positiveValue,negativeValue, newBranch(can be equal to old branch if no pruning was done)
def PostPrune(subTree,validationSet,attr_vals_list,attr_names):
    if(isinstance(subTree,str)):
        finalValuesString = subTree.split("[")[1].split("]")[0]
        positiveValue,negativeValue = finalValuesString.split(",")
        positiveValue = int(positiveValue)
        negativeValue = int(negativeValue)
        return positiveValue,negativeValue,subTree
   
    prunedTree = subTree

    head = subTree[0]
    attributeIndex = 0
    for i in range(len(attr_names)):
        if(head == attr_names[i]):
            attributeIndex = i
            break
    totalPositives = 0
    totalNegatives = 0
    for attributeValueIndex,attributeValue in enumerate(attr_vals_list[attributeIndex]):
        subValidationSet = []
        for validationDatum in validationSet:
            if(validationDatum[attributeIndex] == attributeValue):
                subValidationSet.append(validationDatum)
        positiveValue,negativeValue,newBranch = PostPrune(subTree[attributeValueIndex + 1],subValidationSet,attr_vals_list,attr_names)
        prunedTree[attributeValueIndex + 1] = newBranch
        totalPositives+=positiveValue
        totalNegatives+=negativeValue

    if(len(validationSet) == 0):
        return totalPositives,totalNegatives,subTree

    unprunedAccuracy = GetTreeAccuracy(subTree,validationSet,attr_vals_list,attr_names)
    prunedAccuracy = 0
    if(totalPositives > totalNegatives):
        prunedGuess = attr_vals_list[0]
    else:
        prunedGuess = attr_vals_list[1]
    for validationDatum in validationSet:
        if(validationDatum[-1] == prunedGuess):
            prunedAccuracy+=1
    prunedAccuracy/=len(validationSet)
    if(prunedAccuracy >= unprunedAccuracy):
        return totalPositives,totalNegatives,prunedTree
    else:
        return totalPositives,totalNegatives,subTree

def GetTreeAccuracy(tree,test_data,attr_vals_list,attr_names):
    correctGuessCount = 0
    for datum in test_data:
        result = FollowTree(tree,datum,attr_vals_list,attr_names)
        if(result == datum[-1]):
            correctGuessCount+=1
    return float(correctGuessCount) / len(test_data)

#returns the value of the leaf by following the tree(acc, unacc)
def FollowTree(tree,datum,attr_vals_list,attr_names):
    attributeIndex = 0
    head = tree[0]
    for i in range(len(attr_names)):
        if(head == attr_names[i]):
            attributeIndex = i
            break
    value = datum[attributeIndex]
    valueIndex = 0
    for i in range(len(attr_vals_list[attributeIndex])):
        if(value == attr_vals_list[attributeIndex][i]):
            valueIndex = i
            break
    
    nextTree = tree[valueIndex + 1]
    if(isinstance(nextTree,str)):#if leaf
        return nextTree.split("[")[0]
    elif(isinstance(nextTree[0],str) and len(nextTree) == 1):#prevents errors when nexttree=['acc[2,3]'] etc.  when prepruned
        return nextTree[0].split("[")[0]
    else:
        return FollowTree(nextTree,datum,attr_vals_list,attr_names)

def main():
    with open('hw3_data/dt/data.pkl','rb') as f :
        train_data, test_data, attr_vals_list, attr_names = pickle.load(f)
        #train_data is the values of each entry
        #   consists of [[value1,value2...value7],entry2,entry3...] each value is in order of the attributes given in the pdf
        #attr_vals_list is the values each attribute can get
        #   consists of [[value1,value2..ValueX],otherAttributeValues2,...] each array can be in different length
        #   acc is (+) unacc is (-)
        #attr_names is the names of each attribute in an array


        metrics = ["infoGain","gainRatio","averageGiniIndex"]
        for metric in metrics:
            tree = CreateTree(train_data,attr_vals_list, attr_names,metric)
            print(metric + " tree accuracy= " + str(GetTreeAccuracy(tree,test_data,attr_vals_list,attr_names)))
            dot = Digraph()
            dot.node('1',tree[0])#the root of the tree
            CreateTreeGraph(tree,dot, [1],1)
            try:
                dot.render(filename="DecisionTreeOutputs/" + metric,view=False)
            except Exception:
                pass

        tree = CreatePrePrunedTreeGraph(train_data,attr_vals_list, attr_names)
        print("gain ratio pre pruned tree accuracy= " + str(GetTreeAccuracy(tree,test_data,attr_vals_list,attr_names)))
        tree = CreatePostPrunedTreeGraph(train_data,attr_vals_list,attr_names)
        print("gain ratio post pruned tree accuracy= " + str(GetTreeAccuracy(tree,test_data,attr_vals_list,attr_names)))

if __name__ == '__main__':
    main()
    exit