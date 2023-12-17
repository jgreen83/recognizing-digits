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
from sklearn.decomposition import PCA
from multiprocessing import Pool,Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
from itertools import islice




#reading data
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

print("outside")



testFile = 'spoken+arabic+digit/Test_Arabic_Digit.txt'

wiDigitCountT = 0
digDiffCountT = 0
dataListT = []
sampleListT = []

with open(testFile,'r') as file:
    for line in file:
        if(len(line.strip().split(" ")) == 1):
            if(wiDigitCountT > 110):
                gen = 'female'
            else:
                gen = 'male'
            sampleListT.append(SpeechSample(gen,digDiffCountT,wiDigitCountT,dataListT))
            dataListT = []
            wiDigitCountT += 1
            if(wiDigitCountT > 220):
                wiDigitCountT = 1
                digDiffCountT += 1
        else:
            dataFloat = np.array(line.strip().split(" "),dtype=float)
            dataListT.append(dataFloat)
            
sampleListT = sampleListT[1:]

dataByDigT = [[],[],[],[],[],[],[],[],[],[]]
dataByDigMT = [[],[],[],[],[],[],[],[],[],[]]
dataByDigFT = [[],[],[],[],[],[],[],[],[],[]]
samplesLensByDigT = [[],[],[],[],[],[],[],[],[],[]]
#getting a list of data points for each digit
for sample in sampleListT:
    samplesLensByDigT[sample.digit].append(len(sample.data))
    for row in sample.data:
        dataByDigT[sample.digit].append(row)
        if(sample.gender == 'male'):
            dataByDigMT[sample.digit].append(row)
        elif(sample.gender == 'female'):
            dataByDigFT[sample.digit].append(row)
    
#now each dataByDig[i] is a list of points associated with the digit i
#combine all digits
dAllT = [*dataByDigT[0],*dataByDigT[1],*dataByDigT[2],*dataByDigT[3],*dataByDigT[4],*dataByDigT[5],*dataByDigT[6],*dataByDigT[7],*dataByDigT[8],*dataByDigT[9]]
dAllMT = [*dataByDigMT[0],*dataByDigMT[1],*dataByDigMT[2],*dataByDigMT[3],*dataByDigMT[4],*dataByDigMT[5],*dataByDigMT[6],*dataByDigMT[7],*dataByDigMT[8],*dataByDigMT[9]]
dAllFT = [*dataByDigFT[0],*dataByDigFT[1],*dataByDigFT[2],*dataByDigFT[3],*dataByDigFT[4],*dataByDigFT[5],*dataByDigFT[6],*dataByDigFT[7],*dataByDigFT[8],*dataByDigFT[9]]




def emGmm(data,n_comps):
    model1 = gm(n_components=n_comps,covariance_type='full',tol=0.001,n_init=1,init_params='kmeans')
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

def calc_like(coord,mod,weights):
    weigh = np.array(weights)
    weigh = weigh.flatten()
    probs = np.fromiter(like_func(coord,mod,len(weights)),dtype=float,count=len(weights))
    return np.dot(weigh,probs)

def calc_like_j(coord,j,bmwB):
    for i in range(j):
        yield calc_like(coord,bmwB[i][1],bmwB[i][2])

def like_func(coord,mod,n):
    for k in range(n):
        yield multivariate_normal.pdf(coord,mean=mod.means_[k],cov=mod.covariances_[k])

def calc_likes_per_clust(coord,mod,weights):
    likeList = []
    for k in range(len(weights)):
        likeList.append(multivariate_normal.pdf(coord,mean=mod.means_[k],cov=mod.covariances_[k]))
    return likeList

def make_bmw_comps(data,i):
    bigg,modd,ww = emGmm(data[i],5)
    return [bigg,modd,ww]

def make_bmwL_comps(data,i,k):
    bigg,modd,ww = emGmm(data[i],k)
    return [bigg,modd,ww]

def make_bmwB(data,k,parallel=False):
    bmwB =[]
    for i in range(k):
        bmwB.append([])
    if(parallel):
        pool = Pool(processes=4)
        bmwB = pool.starmap(make_bmw_comps, [(data,val) for val in range(10)])
        pool.close()
    else:
        for i in range(k):
            bmwB[i] = make_bmw_comps(data,i)

    return bmwB

def make_small_like_list(dataByDig,bmwL,bmwB,k):
    oD0Lk = []
    for i in range(len(dataByDig[k])):
        #calc likelihood of dataByDig[k][i]'s MFCC1 being in each of the clusters specified by bmwL[0] (mod for MFCC1)
        likeList = calc_likes_per_clust([[dataByDig[k][i][0]]],bmwL[0][1],bmwL[0][2])
        #count = 0
        calcs = np.fromiter(calc_like_j(dataByDig[k][i],len(likeList),bmwB),dtype=float,count=len(likeList))
        likeList = np.array(likeList)
        likeList.flatten()
    
        oD0Lk.append(np.dot(likeList,calcs))
    return oD0Lk

def make_big_like_list(dataByDig,bmwL,bmwB,parallel=False):
    overallData0Likes = [[],[],[],[],[],[],[],[],[],[]]
    if(parallel):
        pool = Pool(processes=3)
        overallData0Likes = pool.starmap(make_small_like_list,[(dataByDig,bmwL,bmwB,k) for k in range(10)])
    else:
        for k in range(10):
            overallData0Likes[k] = make_small_like_list(dataByDig,bmwL,bmwB,k)

    return overallData0Likes
    

def main(sampleList,dig):
    #start = time.time()
    #my_array = make_bmwB(dig0SortByMFCC1Cluster,parallel=False)
    #end = time.time()
    
    #print(np.array(my_array).shape)
    #print("Normal time: {}\n".format(end-start))

    firstMFCCs = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    #separating out by element (1-13) in each digit (then to cluster into syllables? ? to incorporate some sense of time-dependence?)
    for i in range(len(sampleList)):
        if (sampleList[i].digit == dig): #only doing 0 rn
            for j in range(len(sampleList[i].data[0])):
                #firstMFCCs[j].append([coeff[j] for coeff in sampleList[i].data])
                for coeff in sampleList[i].data:
                    firstMFCCs[j].append([coeff[j]])
    print("done sorting")

    bmwL = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    kkLL = [5, 5, 6, 6, 6, 5, 5, 8, 7, 8]
    for i in range(13):
        bmwL[i] = make_bmwL_comps(firstMFCCs,i,kkLL[dig])

    MFCC1Pred = bmwL[0][1].predict(firstMFCCs[0])
    dig0SortByMFCC1Cluster = []
    for i in range(kkLL[dig]):
        dig0SortByMFCC1Cluster.append([])
    for i in range(len(MFCC1Pred)):
        dig0SortByMFCC1Cluster[MFCC1Pred[i]].append(dataByDig[dig][i])
    print("b time")
    bmwB = make_bmwB(dig0SortByMFCC1Cluster,kkLL[dig],parallel=False)

    #print("starting long part")

    #start_parallel = time.time()
    #overallLikeOfBeingDig = make_big_like_list(dataByDig,bmwL,bmwB,parallel=True)
    #end_parallel = time.time()

    #print("Time based on multiprocessing: {}".format(end_parallel-start_parallel))

    return [bmwL, bmwB]

def formBLL(data,bmwL,bmwB):
    start_parallel = time.time()
    overallLikeOfBeingDig = make_big_like_list(data,bmwL,bmwB,parallel=True)
    end_parallel = time.time()

    print("Time based on multiprocessing: {}".format(end_parallel-start_parallel))

    return overallLikeOfBeingDig


if __name__ == '__main__':
    start = time.time()
    #each element in list "likes" is an array of likelihoods of different samples being of that digit (dig). within this, there is
    #one list for every digit (j), so likes[dig][j][k] is the likelihood that dataByDig[j][k] corresponds to dig
    bmwS = []
    likes = []
    for k in range(10):
        bmwS.append(main(sampleList,k))
        print("starting long part")
        likes.append(formBLL(dataByDigT,bmwS[k][0],bmwS[k][1]))
        print("woohoooooo done w {}".format(k))

    lik = []
    for j in range(10):
        bigJL = []
        for k in range(len(likes[0][j])):
            sampL = []
            for i in range(10):
                sampL.append(likes[i][j][k])
            bigJL.append(sampL)
        lik.append(bigJL)

    splitt = []
    for dig in range(10):
        temp = iter(lik[dig])
        splitt.append([list(islice(temp, 0, ele)) for ele in samplesLensByDigT[dig]])

    maxes = [[],[],[],[],[],[],[],[],[],[]]
    for dig in range(10):
        for samp in splitt[dig]:
            prods = samp[0]
            for i in range(1,len(samp)):
                for j in range(10):
                    prods[j] = prods[j]*samp[i][j]
            maxes[dig].append(prods.index(max(prods)))
    
    corrCt = []
    for dig in range(10):
        corr = 0
        for samp in maxes[dig]:
            if(samp == dig):
                corr += 1
        corrCt.append(corr)

    acc = [] #accuracies by digit
    accOverall = sum(corrCt)/len(sampleListT) #overall accuracy
    for j in range(10):
        acc.append(corrCt[j]/len(samplesLensByDigT[j]))


    
