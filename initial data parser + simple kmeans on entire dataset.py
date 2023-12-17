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
#kmeans! on all data at once
kmeans = KMeans(n_clusters=10,n_init="auto").fit(dAll)
kmeansM = KMeans(n_clusters=10,n_init="auto").fit(dAllM)
kmeansF = KMeans(n_clusters=10,n_init="auto").fit(dAllF)

lab = kmeansM.labels_
labM = kmeansM.labels_
labF = kmeansF.labels_




c0_fin_x = []
c0_fin_y = []
c0_fin_z = []
c1_fin_x = []
c1_fin_y = []
c1_fin_z = []
c2_fin_x = []
c2_fin_y = []
c2_fin_z = []
c3_fin_x = []
c3_fin_y = []
c3_fin_z = []
c4_fin_x = []
c4_fin_y = []
c4_fin_z = []
c5_fin_x = []
c5_fin_y = []
c5_fin_z = []
c6_fin_x = []
c6_fin_y = []
c6_fin_z = []
c7_fin_x = []
c7_fin_y = []
c7_fin_z = []
c8_fin_x = []
c8_fin_y = []
c8_fin_z = []
c9_fin_x = []
c9_fin_y = []
c9_fin_z = []


for i in range(len(kmeans.labels_)):
    if(kmeans.labels_[i] == 0):
        c0_fin_x.append(dAll[i][0])
        c0_fin_y.append(dAll[i][1])
        c0_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 1):
        c1_fin_x.append(dAll[i][0])
        c1_fin_y.append(dAll[i][1])
        c1_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 2):
        c2_fin_x.append(dAll[i][0])
        c2_fin_y.append(dAll[i][1])
        c2_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 3):
        c3_fin_x.append(dAll[i][0])
        c3_fin_y.append(dAll[i][1])
        c3_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 4):
        c4_fin_x.append(dAll[i][0])
        c4_fin_y.append(dAll[i][1])
        c4_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 5):
        c5_fin_x.append(dAll[i][0])
        c5_fin_y.append(dAll[i][1])
        c5_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 6):
        c6_fin_x.append(dAll[i][0])
        c6_fin_y.append(dAll[i][1])
        c6_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 7):
        c7_fin_x.append(dAll[i][0])
        c7_fin_y.append(dAll[i][1])
        c7_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 8):
        c8_fin_x.append(dAll[i][0])
        c8_fin_y.append(dAll[i][1])
        c8_fin_z.append(dAll[i][2])
    elif(kmeans.labels_[i] == 9):
        c9_fin_x.append(dAll[i][0])
        c9_fin_y.append(dAll[i][1])
        c9_fin_z.append(dAll[i][2])

ex = np.linspace(-10,10)
wy = np.linspace(-10,10)
[Ex,Wy] = np.meshgrid(ex,wy)
bigMatC0 = np.empty_like(Ex)

for i in range(len(Ex)):
    for j in range(len(Ex[0])):
        bigMatC0[i][j] = (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[0][0:2],cov=np.cov(c0_fin_x,c0_fin_y)))        
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[1][0:2],cov=np.cov(c1_fin_x,c1_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[2][0:2],cov=np.cov(c2_fin_x,c2_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[3][0:2],cov=np.cov(c3_fin_x,c3_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[4][0:2],cov=np.cov(c4_fin_x,c4_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[5][0:2],cov=np.cov(c5_fin_x,c5_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[6][0:2],cov=np.cov(c6_fin_x,c6_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[7][0:2],cov=np.cov(c7_fin_x,c7_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[8][0:2],cov=np.cov(c8_fin_x,c8_fin_y)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[9][0:2],cov=np.cov(c9_fin_x,c9_fin_y)))
        #bigExList.append(Ex[i][j])

#MFCC 1,2,3 for all 10 digits
M1_0 = []
M2_0 = []
M3_0 = []
M1_1 = []
M2_1 = []
M3_1 = []
M1_2 = []
M2_2 = []
M3_2 = []
M1_3 = []
M2_3 = []
M3_3 = []
M1_4 = []
M2_4 = []
M3_4 = []
M1_5 = []
M2_5 = []
M3_5 = []
M1_6 = []
M2_6 = []
M3_6 = []
M1_7 = []
M2_7 = []
M3_7 = []
M1_8 = []
M2_8 = []
M3_8 = []
M1_9 = []
M2_9 = []
M3_9 = []


for i in dataByDig[0]:
    M1_0.append(i[0])
    M2_0.append(i[1])
    M3_0.append(i[2])
for i in dataByDig[1]:
    M1_1.append(i[0])
    M2_1.append(i[1])
    M3_1.append(i[2])
for i in dataByDig[2]:
    M1_2.append(i[0])
    M2_2.append(i[1])
    M3_2.append(i[2])
for i in dataByDig[3]:
    M1_3.append(i[0])
    M2_3.append(i[1])
    M3_3.append(i[2])
for i in dataByDig[4]:
    M1_4.append(i[0])
    M2_4.append(i[1])
    M3_4.append(i[2])
for i in dataByDig[5]:
    M1_5.append(i[0])
    M2_5.append(i[1])
    M3_5.append(i[2])
for i in dataByDig[6]:
    M1_6.append(i[0])
    M2_6.append(i[1])
    M3_6.append(i[2])
for i in dataByDig[7]:
    M1_7.append(i[0])
    M2_7.append(i[1])
    M3_7.append(i[2])
for i in dataByDig[8]:
    M1_8.append(i[0])
    M2_8.append(i[1])
    M3_8.append(i[2])
for i in dataByDig[9]:
    M1_9.append(i[0])
    M2_9.append(i[1])
    M3_9.append(i[2])



#plotting
plt.scatter(M1_0,M2_0,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 0')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
"""
plt.scatter(M1_1,M2_1,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 1')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_2,M2_2,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 2')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_3,M2_3,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 3')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_4,M2_4,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 4')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_5,M2_5,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 5')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_6,M2_6,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 6')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_7,M2_7,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 7')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_8,M2_8,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 8')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()
plt.scatter(M1_9,M2_9,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC2 for digit 9')
plt.xlabel('MFCC1')
plt.ylabel('MFCC2')
plt.show()

"""

bigMatC0 = np.empty_like(Ex)

for i in range(len(Ex)):
    for j in range(len(Ex[0])):
        bigMatC0[i][j] = (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[0][1:3],cov=np.cov(c0_fin_y,c0_fin_z)))        
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[1][1:3],cov=np.cov(c1_fin_y,c1_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[2][1:3],cov=np.cov(c2_fin_y,c2_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[3][1:3],cov=np.cov(c3_fin_y,c3_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[4][1:3],cov=np.cov(c4_fin_y,c4_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[5][1:3],cov=np.cov(c5_fin_y,c5_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[6][1:3],cov=np.cov(c6_fin_y,c6_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[7][1:3],cov=np.cov(c7_fin_y,c7_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[8][1:3],cov=np.cov(c8_fin_y,c8_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=kmeans.cluster_centers_[9][1:3],cov=np.cov(c9_fin_y,c9_fin_z)))



#plotting
plt.scatter(M2_0,M3_0,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 0')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()

"""
plt.scatter(M2_1,M3_1,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 1')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_2,M3_2,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 2')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_3,M3_3,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 3')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_4,M3_4,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 4')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_5,M3_5,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 5')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_6,M3_6,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 6')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_7,M3_7,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 7')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_8,M3_8,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 8')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M2_9,M3_9,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC2 vs MFCC3 for digit 9')
plt.xlabel('MFCC2')
plt.ylabel('MFCC3')
plt.show()
"""

bigMatC0 = np.empty_like(Ex)

for i in range(len(Ex)):
    for j in range(len(Ex[0])):
        bigMatC0[i][j] = (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][2]],cov=np.cov(c0_fin_x,c0_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][2]],cov=np.cov(c1_fin_x,c1_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[2][0],kmeans.cluster_centers_[2][2]],cov=np.cov(c2_fin_x,c2_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[3][0],kmeans.cluster_centers_[3][2]],cov=np.cov(c3_fin_x,c3_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[4][0],kmeans.cluster_centers_[4][2]],cov=np.cov(c4_fin_x,c4_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[5][0],kmeans.cluster_centers_[5][2]],cov=np.cov(c5_fin_x,c5_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[6][0],kmeans.cluster_centers_[6][2]],cov=np.cov(c6_fin_x,c6_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[7][0],kmeans.cluster_centers_[7][2]],cov=np.cov(c7_fin_x,c7_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[8][0],kmeans.cluster_centers_[8][2]],cov=np.cov(c8_fin_x,c8_fin_z)))
        bigMatC0[i][j] += (multivariate_normal.pdf([Ex[i][j],Wy[i][j]],mean=[kmeans.cluster_centers_[9][0],kmeans.cluster_centers_[9][2]],cov=np.cov(c9_fin_x,c9_fin_z)))


#plotting
plt.scatter(M1_0,M3_0,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 0')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()

"""
plt.scatter(M1_1,M3_1,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 1')
plt.xlabel('MFCC=1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_2,M3_2,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 2')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_3,M3_3,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 3')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_4,M3_4,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 4')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_5,M3_5,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 5')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_6,M3_6,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 6')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_7,M3_7,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 7')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_8,M3_8,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 8')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
plt.scatter(M1_9,M3_9,alpha=.4)
plt.contour(Ex,Wy,bigMatC0)
plt.title('MFCC1 vs MFCC3 for digit 9')
plt.xlabel('MFCC1')
plt.ylabel('MFCC3')
plt.show()
"""


        
"""
modes = [0,0,0,0,0,0,0,0,0,0]
modes[0] = mode(lab[0:len(dataByDig[0])])
modes[1] = mode(lab[len(dataByDig[0]):len(dataByDig[0])+len(dataByDig[1])])
modes[2] = mode(lab[len(dataByDig[0])+len(dataByDig[1]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])])
modes[3] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])])
modes[4] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])])
modes[5] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])])
modes[6] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])])
modes[7] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])+len(dataByDig[7])])
modes[8] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])+len(dataByDig[7]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])+len(dataByDig[7])+len(dataByDig[8])])
modes[9] = mode(lab[len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])+len(dataByDig[7])+len(dataByDig[8]):len(dataByDig[0])+len(dataByDig[1])+len(dataByDig[2])+len(dataByDig[3])+len(dataByDig[4])+len(dataByDig[5])+len(dataByDig[6])+len(dataByDig[7])+len(dataByDig[8])+len(dataByDig[9])])

modesM = [0,0,0,0,0,0,0,0,0,0]
modesM[0] = mode(lab[0:len(dataByDigM[0])])
modesM[1] = mode(lab[len(dataByDigM[0]):len(dataByDigM[0])+len(dataByDigM[1])])
modesM[2] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])])
modesM[3] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])])
modesM[4] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])])
modesM[5] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])])
modesM[6] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])])
modesM[7] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])+len(dataByDigM[7])])
modesM[8] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])+len(dataByDigM[7]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])+len(dataByDigM[7])+len(dataByDigM[8])])
modesM[9] = mode(lab[len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])+len(dataByDigM[7])+len(dataByDigM[8]):len(dataByDigM[0])+len(dataByDigM[1])+len(dataByDigM[2])+len(dataByDigM[3])+len(dataByDigM[4])+len(dataByDigM[5])+len(dataByDigM[6])+len(dataByDigM[7])+len(dataByDigM[8])+len(dataByDigM[9])])

"""


"""
for i in range(len(kmeans.labels_)):
    if(kmeans.labels_[i] == 0):
        c1_fin.append(cAll[i][0])
        c1_fin_y.append(cAll[i][1])
    elif(kmeans.labels_[i] == 1):
        c2_fin_x.append(cAll[i][0])
        c2_fin_y.append(cAll[i][1])
    elif(kmeans.labels_[i] == 2):
        c3_fin_x.append(cAll[i][0])
        c3_fin_y.append(cAll[i][1])
    elif(kmeans.labels_[i] == 3):
        c4_fin_x.append(cAll[i][0])
        c4_fin_y.append(cAll[i][1])
"""



#getting first three MECC's from one utterance of each digit
sample0 = sampleList[0]
sample1 = sampleList[660]
sample2 = sampleList[2*660]
sample3 = sampleList[3*660]
sample4 = sampleList[4*660]
sample5 = sampleList[5*660]
sample6 = sampleList[6*660]
sample7 = sampleList[7*660]
sample8 = sampleList[8*660]
sample9 = sampleList[9*660]

M1_0 = []
M2_0 = []
M3_0 = []
M1_1 = []
M2_1 = []
M3_1 = []
M1_2 = []
M2_2 = []
M3_2 = []
M1_3 = []
M2_3 = []
M3_3 = []
M1_4 = []
M2_4 = []
M3_4 = []
M1_5 = []
M2_5 = []
M3_5 = []
M1_6 = []
M2_6 = []
M3_6 = []
M1_7 = []
M2_7 = []
M3_7 = []
M1_8 = []
M2_8 = []
M3_8 = []
M1_9 = []
M2_9 = []
M3_9 = []

for i in sample0.data:
    M1_0.append(i[0])
    M2_0.append(i[1])
    M3_0.append(i[2])
for i in sample1.data:
    M1_1.append(i[0])
    M2_1.append(i[1])
    M3_1.append(i[2])
for i in sample2.data:
    M1_2.append(i[0])
    M2_2.append(i[1])
    M3_2.append(i[2])
for i in sample3.data:
    M1_3.append(i[0])
    M2_3.append(i[1])
    M3_3.append(i[2])
for i in sample4.data:
    M1_4.append(i[0])
    M2_4.append(i[1])
    M3_4.append(i[2])
for i in sample5.data:
    M1_5.append(i[0])
    M2_5.append(i[1])
    M3_5.append(i[2])
for i in sample6.data:
    M1_6.append(i[0])
    M2_6.append(i[1])
    M3_6.append(i[2])
for i in sample7.data:
    M1_7.append(i[0])
    M2_7.append(i[1])
    M3_7.append(i[2])
for i in sample8.data:
    M1_8.append(i[0])
    M2_8.append(i[1])
    M3_8.append(i[2])
for i in sample9.data:
    M1_9.append(i[0])
    M2_9.append(i[1])
    M3_9.append(i[2])


#plotting
plt.plot(M1_0,label='MFCC1')
plt.plot(M2_0,label='MFCC2')
plt.plot(M3_0,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 0')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_1,label='MFCC1')
plt.plot(M2_1,label='MFCC2')
plt.plot(M3_1,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 1')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_2,label='MFCC1')
plt.plot(M2_2,label='MFCC2')
plt.plot(M3_2,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 2')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_3,label='MFCC1')
plt.plot(M2_3,label='MFCC2')
plt.plot(M3_3,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 3')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_4,label='MFCC1')
plt.plot(M2_4,label='MFCC2')
plt.plot(M3_4,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 4')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_5,label='MFCC1')
plt.plot(M2_5,label='MFCC2')
plt.plot(M3_5,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 5')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_6,label='MFCC1')
plt.plot(M2_6,label='MFCC2')
plt.plot(M3_6,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 6')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_7,label='MFCC1')
plt.plot(M2_7,label='MFCC2')
plt.plot(M3_7,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 7')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_8,label='MFCC1')
plt.plot(M2_8,label='MFCC2')
plt.plot(M3_8,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 8')
plt.legend(loc='upper right')
plt.show()

plt.plot(M1_9,label='MFCC1')
plt.plot(M2_9,label='MFCC2')
plt.plot(M3_9,label='MFCC3')
plt.xlabel('analysis window')
plt.ylabel('MFCC value')
plt.title('MFCC1,2,3 for utterance of 9')
plt.legend(loc='upper right')
plt.show()


"""

plt.clf()
plt.scatter(M1_0,M2_0,label='M1 vs M2')
plt.scatter(M1_0,M3_0,label='M1 vs M3')
plt.scatter(M2_0,M3_0,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 0')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_1,M2_1,label='M1 vs M2')
plt.scatter(M1_1,M3_1,label='M1 vs M3')
plt.scatter(M2_1,M3_1,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 1')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_2,M2_2,label='M1 vs M2')
plt.scatter(M1_2,M3_2,label='M1 vs M3')
plt.scatter(M2_2,M3_2,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 2')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_3,M2_3,label='M1 vs M2')
plt.scatter(M1_3,M3_3,label='M1 vs M3')
plt.scatter(M2_3,M3_3,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 3')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_4,M2_4,label='M1 vs M2')
plt.scatter(M1_4,M3_4,label='M1 vs M3')
plt.scatter(M2_4,M3_4,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 4')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_5,M2_5,label='M1 vs M2')
plt.scatter(M1_5,M3_5,label='M1 vs M3')
plt.scatter(M2_5,M3_5,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 5')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_6,M2_6,label='M1 vs M2')
plt.scatter(M1_6,M3_6,label='M1 vs M3')
plt.scatter(M2_6,M3_6,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 6')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_7,M2_7,label='M1 vs M2')
plt.scatter(M1_7,M3_7,label='M1 vs M3')
plt.scatter(M2_7,M3_7,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 7')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_8,M2_8,label='M1 vs M2')
plt.scatter(M1_8,M3_8,label='M1 vs M3')
plt.scatter(M2_8,M3_8,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 8')
plt.legend()
plt.show()

plt.clf()
plt.scatter(M1_9,M2_9,label='M1 vs M2')
plt.scatter(M1_9,M3_9,label='M1 vs M3')
plt.scatter(M2_9,M3_9,label='M2 vs M3')
plt.title('MFCC scatter plot for utterance of digit 9')
plt.legend()
plt.show()



"""


    







