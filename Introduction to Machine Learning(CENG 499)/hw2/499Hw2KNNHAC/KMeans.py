import matplotlib.pyplot as plt
import numpy as np
import random

#returns squared magnitude of x
#can be used as a squared distance function between x and y by passing x-y
def SquaredMagnitude(x):
    return np.inner(x,x)

def ApplyKMeans(clustering,KValues=range(1,11),plotFinalClusters=False,savePlot=False,plotName=""):#range(1,11) for [1,10] clusters
    if(plotFinalClusters):
        minObjectiveFunction = float("inf")
        minObjectiveFunctionAssignedClusterIDs = [0] * clustering.shape[0]
        if(len(KValues) != 1):
            raise Exception("KValues needs to be of length 1 in order to plot the results")
            exit()
    if(savePlot and plotName == ""):
        raise Exception("give a proper plot name to plot the results")
        exit()

    minXValue = np.amin(clustering,axis=0)[0]
    minYValue = np.amin(clustering,axis=0)[1]
    maxXValue = np.amax(clustering,axis=0)[0]
    maxYValue = np.amax(clustering,axis=0)[1]
    
    trialCount = 1
    objectiveFunctionAverageValues = [0] * len(KValues)#K=[1,10]
    
    for K in KValues:
        clusterCenters = [(0,0)] * K
        for trialID in range(trialCount):#re-try trial count times
            for clusterID in range(K):#initialize te clusters
                clusterCenters[clusterID] = random.uniform(minXValue,maxXValue),random.uniform(minYValue,maxYValue)
        
            #apply kmeans until the clusters don't change positions
            #i.e until obj function doesn't change
            assignedClusterIDs = [0] * clustering.shape[0]
            objectiveFunction = float("inf")
            while True:
                newObjectiveFunction = 0
                #assign points to clusters
                for dataIndex in range(clustering.shape[0]):
                    minSquaredDistance = float("inf")
                    for clusterID in range(K):
                        squaredDistance = SquaredMagnitude(clustering[dataIndex] - clusterCenters[clusterID])
                        if(squaredDistance < minSquaredDistance):
                            minSquaredDistance = squaredDistance
                            assignedClusterIDs[dataIndex] = clusterID
                    #update the new objective function
                    newObjectiveFunction+=minSquaredDistance
    
                if(objectiveFunction == newObjectiveFunction):#obj function didn't change, stop
                    if(plotFinalClusters and objectiveFunction < minObjectiveFunction):#save the current assigned cluster id's for plotting
                        minObjectiveFunction = objectiveFunction
                        minObjectiveFunctionAssignedClusterIDs = assignedClusterIDs
                    objectiveFunctionAverageValues[K - KValues[0]]+=objectiveFunction / trialCount#add the current objective function as average objective function
                    break
                objectiveFunction = newObjectiveFunction
                #update cluster centers
                clusterXSums = [0] * K
                clusterYSums = [0] * K
                for dataIndex in range(clustering.shape[0]):
                    clusterXSums[assignedClusterIDs[dataIndex]]+=clustering[dataIndex][0]
                    clusterYSums[assignedClusterIDs[dataIndex]]+=clustering[dataIndex][1]
                for clusterID in range(K):
                    assignedDataCount = assignedClusterIDs.count(clusterID)
                    if(assignedDataCount == 0):
                        continue
                    else:
                        clusterCenters[clusterID] = clusterXSums[clusterID] / assignedDataCount,clusterYSums[clusterID] / assignedDataCount

    if(plotFinalClusters):
        finalClusters = [np.zeros(1)] * KValues[0]
        finalClusterIndexes = [0] * KValues[0]#used for keeping the index that is not set to a value
        for i in range(KValues[0]):
            finalClusters[i] = np.zeros((minObjectiveFunctionAssignedClusterIDs.count(i),2))
            #need to select data with minObjectiveFunctionAssignedClusterIDs=i
        for index,datum in enumerate(clustering):
            clusterID = minObjectiveFunctionAssignedClusterIDs[index]
            clusterIndex = finalClusterIndexes[clusterID]
            finalClusterIndexes[clusterID]+=1
            finalClusters[clusterID][clusterIndex][0] = datum[0]
            finalClusters[clusterID][clusterIndex][1] = datum[1]

        plt.title(plotName)
        plt.xlabel("X")
        plt.ylabel("Y")
        for i in range(KValues[0]):
            plt.scatter(finalClusters[i][:,0],finalClusters[i][:,1],label="cluster" + str(i))
        plt.legend()
        if(savePlot):
            plt.savefig("../KMeans/FinalClusters - " + plotName + ".png")
            plt.close()
        else:
            plt.show()
    else:
        plt.xlabel("KValues")
        plt.ylabel("average objective function")
        plt.axis([0,len(KValues) + 1,0,max(objectiveFunctionAverageValues) + 100])
        plt.plot(KValues,objectiveFunctionAverageValues,label="average objective function value at K clusters", marker='o',fillstyle="none")
        plt.legend()
        if(savePlot):
            plt.savefig("../KMeans/ObjectiveFunctionValues - " + plotName + ".png")
            plt.close()
        else:
            plt.show()

def main():
    clustering1 = np.load('hw2_data/kmeans/clustering1.npy')
    clustering2 = np.load('hw2_data/kmeans/clustering2.npy')
    clustering3 = np.load('hw2_data/kmeans/clustering3.npy')
    clustering4 = np.load('hw2_data/kmeans/clustering4.npy')

    #plot objective function vs K
    ApplyKMeans(clustering1,range(1,11),False,True,"clustering1")
    ApplyKMeans(clustering2,range(1,11),False,True,"clustering2")
    ApplyKMeans(clustering3,range(1,11),False,True,"clustering3")
    ApplyKMeans(clustering4,range(1,11),False,True,"clustering4")
    
    ApplyKMeans(clustering1,range(2,3),True,True,"clustering1")#K=2
    ApplyKMeans(clustering2,range(3,4),True,True,"clustering2")#K=3
    ApplyKMeans(clustering3,range(4,5),True,True,"clustering3")#K=4
    ApplyKMeans(clustering4,range(5,6),True,True,"clustering4")#K=5
if __name__ == '__main__':
    main()
    exit