import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as gm
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
        dataByDig[sample.digit].append(row)
        if(sample.gender == 'male'):
            dataByDigM[sample.digit].append(row)
        elif(sample.gender == 'female'):
            dataByDigF[sample.digit].append(row)
    
#now each dataByDig[i] is a list of points associated with the digit i
#combine all digits
dAll = [*dataByDig[0],*dataByDig[1],*dataByDig[2],*dataByDig[3],*dataByDig[4],*dataByDig[5],*dataByDig[6],*dataByDig[7],*dataByDig[8],*dataByDig[9]]
dAllM = [*dataByDigM[0],*dataByDigM[1],*dataByDigM[2],*dataByDigM[3],*dataByDigM[4],*dataByDigM[5],*dataByDigM[6],*dataByDigM[7],*dataByDigM[8],*dataByDigM[9]]
dAllF = [*dataByDigF[0],*dataByDigF[1],*dataByDigF[2],*dataByDigF[3],*dataByDigF[4],*dataByDigF[5],*dataByDigF[6],*dataByDigF[7],*dataByDigF[8],*dataByDigF[9]]

#function for expectation maximization
def emGmm(data,n_comps):
    model1 = gm(n_components=n_comps,covariance_type='full',tol=0.001,n_init=1,init_params='random_from_data')
    ff = model1.fit(data)
    fitt = model1.fit_predict(data)

    numOcc = np.empty((n_comps,1))
    for i in range(1,n_comps):
        numOcc[i] = np.count_nonzero(fitt == i)
    numOcc[0] = len(fitt) - sum(numOcc[1:n_comps])

    bigList = []
    for i in range(n_comps):
        bigList.append([])
        for j in range(len(data[0])):
            bigList[i].append([])

    bigSepList = []
    for i in range(len(data[0])):
        bigSepList.append([])

    for i in range(len(data)):
        for j in range(len(data[0])):
            bigList[fitt[i]][j].append(data[i][j])

    weights = numOcc/len(fitt)

    return bigList,model1,weights

def plot_cont(bL,mod,weights,firstInd,secondInd,plotTitle):
    ex = np.linspace(-20,20)
    wy = np.linspace(-20,20)
    [Ex,Wy] = np.meshgrid(ex,wy)

    bigMat = np.empty_like(Ex)

    for i in range(len(Ex)):
        for j in range(len(Ex[0])):
            constrMatVal = 0
            for k in range(len(weights)):
                constrMatVal += weights[k]*(multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[mod.means_[k][firstInd],mod.means_[k][secondInd]],cov=[[mod.covariances_[k][firstInd][firstInd],mod.covariances_[k][firstInd][secondInd]],[mod.covariances_[k][secondInd][firstInd],mod.covariances_[k][secondInd][secondInd]]]))
            bigMat[i][j] = constrMatVal

    plt.clf()
    for i in range(len(bL)):
        plt.scatter(bL[i][firstInd],bL[i][secondInd],alpha=.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(plotTitle)
    plt.contour(Ex,Wy,bigMat)

    plt.show()

def calc_like(coord,mod,weights):
    constrMatVal = 0
    for k in range(len(weights)):
        constrMatVal += weights[k]*(multivariate_normal.pdf(coord,mean=mod.means_[k],cov=mod.covariances_[k]))
    return constrMatVal

#define number of clusters for each digit
numC = [2,2,2,3,3,2,2,2,4,2]
allMods = []

for i in range(10):
    bigg,modd,ww = emGmm(dataByDigF[i],10)
    allMods.append([bigg,modd,ww])
    """
    avgLike = [0,0,0,0,0,0,0,0,0,0]
    for j in range(10):
        for k in range(len(dataByDigF[j])):
            avgLike[j] += calc_like(dataByDigF[j][k],modd,ww)
        avgLike[j] = avgLike[j]/len(dataByDigF[j])
    print("digit " + str(i))
    print(avgLike)
    plot_cont(bigg,modd,ww,0,1,"digit " + str(i) + " MFCC1 vs MFCC2")
    plot_cont(bigg,modd,ww,0,2,"digit " + str(i) + " MFCC1 vs MFCC3")
    plot_cont(bigg,modd,ww,1,2,"digit " + str(i) + " MFCC2 vs MFCC3")
    """
        


maxInds = [[],[],[],[],[],[],[],[],[],[]]
for i in range(10):
    for k in range(len(dataByDigF[i])):
        likeL = []
        for j in range(10):
            likeL.append(calc_like(dataByDigF[i][k],allMods[j][1],allMods[j][2]))
        maxInds[i].append(likeL.index(max(likeL)))


                   


indsCounts = [[],[],[],[],[],[],[],[],[],[]]
for i in range(10):
    for j in range(10):
        indsCounts[i].append(maxInds[i].count(j))


corrCt = 0
totalCt = 0
for i in range(10):
    for j in range(10):
        totalCt += indsCounts[i][j]
    corrCt += indsCounts[i][i]
percentCorr = corrCt/totalCt





    
    

