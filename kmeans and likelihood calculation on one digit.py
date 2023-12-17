import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from statistics import mode



class SpeechSample:
    def __init__(self,gender,digit,index,data):
        self.gender = gender
        self.digit = digit
        self.index = index
        self.data = data

trainingFile = 'spoken+arabic+digit/Train_Arabic_Digit.txt'

wiDigitCount = 0
digDiffCount = 0
dataList = []
sampleList = []

with open(trainingFile,'r') as file:
    for line in file:
        if(len(line.strip().split(" ")) == 1):
            if(wiDigitCount > 330):
                gen = 'female'
            else:
                gen = 'male'
            sampleList.append(SpeechSample(gen,digDiffCount,wiDigitCount,dataList))
            dataList = []
            wiDigitCount += 1
            if(wiDigitCount > 660):
                wiDigitCount = 1
                digDiffCount += 1
        else:
            dataFloat = np.array(line.strip().split(" "),dtype=float)
            dataList.append(dataFloat)
            
sampleList = sampleList[1:]

dataByDig = [[],[],[],[],[],[],[],[],[],[]]
dataByDigM = [[],[],[],[],[],[],[],[],[],[]]
dataByDigF = [[],[],[],[],[],[],[],[],[],[]]
#getting a list of data points for each digit
for sample in sampleList:
    for row in sample.data:
        #dataByDig[sample.digit].append(row)
        if(sample.gender == 'male'):
            dataByDig[sample.digit].append(row)
        elif(sample.gender == 'female'):
            dataByDigF[sample.digit].append(row)
    
#now each dataByDig[i] is a list of points associated with the digit i
#combine all digits
dAll = [*dataByDig[0],*dataByDig[1],*dataByDig[2],*dataByDig[3],*dataByDig[4],*dataByDig[5],*dataByDig[6],*dataByDig[7],*dataByDig[8],*dataByDig[9]]
dAllM = [*dataByDigM[0],*dataByDigM[1],*dataByDigM[2],*dataByDigM[3],*dataByDigM[4],*dataByDigM[5],*dataByDigM[6],*dataByDigM[7],*dataByDigM[8],*dataByDigM[9]]
dAllF = [*dataByDigF[0],*dataByDigF[1],*dataByDigF[2],*dataByDigF[3],*dataByDigF[4],*dataByDigF[5],*dataByDigF[6],*dataByDigF[7],*dataByDigF[8],*dataByDigF[9]]
#kmeans!
kmeans = KMeans(n_clusters=3,n_init="auto").fit(dataByDig[3])
#kmeansM = KMeans(n_clusters=10,n_init="auto").fit(dAllM)
#kmeansF = KMeans(n_clusters=10,n_init="auto").fit(dAllF)

lab = kmeans.labels_
#labM = kmeansM.labels_
#labF = kmeansF.labels_

clustered = [[],[],[]]
for i in range(len(kmeans.labels_)):
    clustered[kmeans.labels_[i]].append(dataByDig[3][i])

#data in each cluster separated by dimension (for cov)
c013 = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
c113 = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
c213 = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(clustered[0])):
  for j in range(len(clustered[0][0])):
      c013[j].append(clustered[0][i][j])
for i in range(len(clustered[1])):
  for j in range(len(clustered[1][0])):
      c113[j].append(clustered[1][i][j])
for i in range(len(clustered[2])):
  for j in range(len(clustered[2][0])):
      c213[j].append(clustered[2][i][j])

c0Cov = np.cov(c013)
c1Cov = np.cov(c113)
c2Cov = np.cov(c213)

"""
#printing mixture components
print("mixture component probabilities: ")
print("p1 = " + str(len(clustered[0])/len(dataByDig[3])))
print("p2 = " + str(len(clustered[1])/len(dataByDig[3])))
print("p3 = " + str(len(clustered[2])/len(dataByDig[3])))

print("mixture component means: ")
print("m1 = " + str(kmeans.cluster_centers_[0]))
print("m2 = " + str(kmeans.cluster_centers_[1]))
print("m3 = " + str(kmeans.cluster_centers_[2]))

print("mixture component covariances: ")
print("S1 = " + str(c0Cov))
print("S2 = " + str(c1Cov))
print("S3 = " + str(c2Cov))
"""

#calculating likelihood of an utterance
zLikes = []
for i in range(len(dataByDig[0])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[0][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[0][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[0][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    zLikes.append(like)

oLikes = []
for i in range(len(dataByDig[1])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[1][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[1][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[1][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    oLikes.append(like)

twLikes = []
for i in range(len(dataByDig[2])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[2][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[2][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[2][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    twLikes.append(like)

thLikes = []
for i in range(len(dataByDig[3])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[3][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[3][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[3][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    thLikes.append(like)

foLikes = []
for i in range(len(dataByDig[4])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[4][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[4][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[4][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    foLikes.append(like)

fiLikes = []
for i in range(len(dataByDig[5])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[5][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[5][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[5][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    fiLikes.append(like)

siLikes = []
for i in range(len(dataByDig[6])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[6][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[6][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[6][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    siLikes.append(like)

seLikes = []
for i in range(len(dataByDig[7])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[7][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[7][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[7][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    seLikes.append(like)

eLikes = []
for i in range(len(dataByDig[8])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[8][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[8][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[8][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    eLikes.append(like)

nLikes = []
for i in range(len(dataByDig[9])):
    like = ((len(clustered[0])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[9][i],mean=kmeans.cluster_centers_[0],cov=c0Cov))) \
           + ((len(clustered[1])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[9][i],mean=kmeans.cluster_centers_[1],cov=c1Cov))) \
           + ((len(clustered[2])/len(dataByDig[3]))*(multivariate_normal.pdf(dataByDig[9][i],mean=kmeans.cluster_centers_[2],cov=c2Cov)))
    nLikes.append(like)



#plotting pdfs
"""
x, bins, p=plt.hist(np.log(zLikes), density=True, bins = 100)
for item in p:
    item.set_height(item.get_height()/sum(x))
#plt.title('pdf of likelihoods of utterances of 0 using estimated GMM')
#plt.xlabel('likelihood')
#plt.ylabel('f(likelihood)')
plt.show()



x, bins, p=plt.hist(np.log(oLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(twLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(thLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(foLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(fiLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(siLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(seLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(eLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()

x, bins, p=plt.hist(np.log(nLikes), density=True,bins=100)
for item in p:
    item.set_height(item.get_height()/sum(x))
plt.show()


"""


