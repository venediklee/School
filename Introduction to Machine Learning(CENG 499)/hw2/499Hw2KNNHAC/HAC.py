import matplotlib.pyplot as plt
import numpy as np

#returns squared magnitude of x
#can be used as a squared distance function between x and y by passing x-y
def SquaredMagnitude(x):
    return np.inner(x,x)

def SingleLinkage(cluster1, cluster2):
    minSquaredDistance = float("inf")#min for cluster1-cluster2
    for datum1 in cluster1:
        for datum2 in cluster2:
            squaredDistance = SquaredMagnitude(datum1 - datum2)
            if(squaredDistance < minSquaredDistance):
                minSquaredDistance = squaredDistance
    return minSquaredDistance

def CompleteLinkage(cluster1, cluster2):
    maxSquaredDistance = -1#max for cluster1-cluster2
    for datum1 in cluster1:
        for datum2 in cluster2:
            squaredDistance = SquaredMagnitude(datum1 - datum2)
            if(squaredDistance > maxSquaredDistance):
                maxSquaredDistance = squaredDistance
    return maxSquaredDistance

def AverageLinkage(cluster1, cluster2):
    squaredDistanceSum = 0
    for datum1 in cluster1:
        for datum2 in cluster2:
            squaredDistanceSum += SquaredMagnitude(datum1 - datum2)

    return squaredDistanceSum / (len(cluster1) * len(cluster2))

def CentroidLinkage(cluster1, cluster2):
    center1 = np.zeros(2)
    center2 = np.zeros(2)
    for datum1 in cluster1:
        center1+=datum1
    for datum2 in cluster2:
        center2+=datum2
    center1/=len(cluster1)
    center2/=len(cluster2)

    return SquaredMagnitude(center1 - center2)

def ApplyHAC(dataIndex,clusterCount,linkage="single",savePlot=False):
    data = np.load('hw2_data/hac/data' + str(dataIndex) + '.npy')
    #convert data to my preffered topology
    clusters = list()
    for datum in data:#fill the clusters list with arrays of size 1, containing all data
        clusters.append([datum])#[ [cluster1=[x1,y1],[x2,y2]...], [cluster2], [cluster3]....]
    
    while len(clusters) != clusterCount:#cluster until there are "clusterCount" clusters left
        #compare each cluster according to distance metric
        minSquaredDistance = float("inf")#min for all clusters in this iteration
        minDistanceIndexes = -1,-1#indexes of clusters
        for index1,cluster1 in enumerate(clusters):
            for index2,cluster2 in enumerate(clusters[index1 + 1:],start=index1 + 1):#no need to check the previous clusters
                #for each datum-datum in cluster1-cluster2, check the distances
                if(linkage == "single"):
                    squaredDistance = SingleLinkage(cluster1, cluster2)
                elif(linkage == "complete"):
                    squaredDistance = CompleteLinkage(cluster1, cluster2)
                elif(linkage == "average"):
                    squaredDistance = AverageLinkage(cluster1, cluster2)
                elif(linkage == "centroid"):
                    squaredDistance = CentroidLinkage(cluster1, cluster2)
                else:
                    raise Exception("wrong linkage specified")
                    exit()

                if(squaredDistance < minSquaredDistance):
                        minSquaredDistance = squaredDistance
                        minDistanceIndexes = index1,index2
        #merge the clusters with the shortest distance
        for datum in clusters[minDistanceIndexes[1]]:
            clusters[minDistanceIndexes[0]].append(datum)
        #remove the merged cluster
        clusters.pop(minDistanceIndexes[1])
    
    
    #convert data to numpy array for easy plotting
    finalClusters = [np.zeros(1)] * clusterCount
    for i in range(clusterCount):
        finalClusters[i] = np.zeros((len(clusters[i]),2))
        for index,datum in enumerate(clusters[i]):
            finalClusters[i][index][0] = datum[0]
            finalClusters[i][index][1] = datum[1]
    
    plt.title("data" + str(dataIndex) + " - " + linkage + " linkage")
    plt.xlabel("X")
    plt.ylabel("Y")
    for i in range(clusterCount):
        plt.scatter(finalClusters[i][:,0],finalClusters[i][:,1],label="cluster" + str(i))
    plt.legend()
    if(savePlot):
        plt.savefig("../HAC/data" + str(dataIndex) + " - " + linkage + " linkage.png")
        plt.close()
    else:
        plt.show()

def main():
    linkageTypes = ["single","complete","average","centroid"]

    for dataIndex in range(1,4):#data1-data2-data3
        for linkage in linkageTypes:
            ApplyHAC(dataIndex,2,linkage,True)

    for linkage in linkageTypes:#data4
            ApplyHAC(4,4,linkage,True)

if __name__ == '__main__':
    main()
    exit
