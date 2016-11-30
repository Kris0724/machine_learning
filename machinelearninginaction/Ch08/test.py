
import regression

from numpy import *
#xArr, yArr = regression.loadDataSet('ex0.txt')
xArr, yArr = regression.loadDataSet('ex1.txt')
#xArr, yArr = regression.loadDataSet('map_data2.txt')
#print xArr
#print
#print yArr

#ws = regression.standRegres(xArr, yArr)
#print ws
#print

#xMat = mat(xArr)
#yMat = mat(yArr)
#yHat = xMat*ws

#abX, abY = regression.loadDataSet('abalone.txt')
#ws = regression.standRegres(abX[0:99], abY[0:99])
#m = shape(abX)[0]
#print m
#yHat = zeros(99)
#for i in range(99):
#    yHat[i] = abX[i]*ws
#print yHati
#TRAIN ERROR
#print regression.rssError(abY[0:99], yHat.T)
#yHat2 = zeros(99)
#for i in range(99):
#    yHat2[i] = abX[100+i]*ws
#TEST ERROR
#print regression.rssError(abY[100:199], yHat2.T)
#exit(0)

'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1], yHat)
plt.show()
'''

#print yHat
#print corrcoef(yHat.T, yMat)

#print yArr[0]
#print
#print "yArr[0]=",yArr[0]
#print "yHat[0]=",regression.lwlr(xArr[0], xArr, yArr, 1.0)
#print regression.lwlr(xArr[0], xArr, yArr, 0.01)
#print regression.lwlrTest(xArr, xArr, yArr, 0.003)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 1.0)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
'''
xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1], yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()
'''


#xArr, yArr = regression.loadDataSet('map_data.txt')
#print xArr
#print
#print yArr
#print 
#ws = regression.standRegres(xArr, yArr)
#print ws
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.3)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.001)
#print regression.rssError(yArr[:], yHat.T)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
#print regression.rssError(yArr[:], yHat.T)
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)
#print regression.rssError(yArr[:], yHat.T)
#print yHat
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.1)
#print regression.rssError(yArr[:], yHat.T)
#yHat = regression.lwlrTest(xArr, xArr, yArr, 1)
print regression.rssError(yArr[:], yHat.T)
#exit(0)

#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)
#print
#yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
#print


xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1], yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()


#yHat1 = regression.lwlrTest(xArr, xArr, yArr, 1.0)
#print regression.rssError(yArr[:], yHat1.T)
#print
#yHat05 = regression.lwlrTest(xArr, xArr, yArr, 0.5)
#print regression.rssError(yArr[:], yHat05.T)
#print
#yHat01 = regression.lwlrTest(xArr, xArr, yArr, 0.18)
#print regression.rssError(yArr[:], yHat01.T)

#print regression.ridgeRegres(xArr, yArr)

#xMat = mat(xArr)
#yMat = mat(yArr)
#yHat = xMat*ws
#print yHat


'''
abX, abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
#print yHat01

#TRAIN ERROR
print regression.rssError(abY[0:99], yHat01.T)
print regression.rssError(abY[0:99], yHat1.T)
print regression.rssError(abY[0:99], yHat10.T)

#abX, abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
#print yHat01

#TEST ERROR
print regression.rssError(abY[100:199], yHat01.T)
print regression.rssError(abY[100:199], yHat1.T)
print regression.rssError(abY[100:199], yHat10.T)
'''



#abX, abY = regression.loadDataSet('abalone.txt')
#print abX
#print
#print abY
#print abX[0:2]

#ws = regression.standRegres(abX,abY)
#print abY[1]
#print
#print ws
#print
#x = mat(abX[1])
#print x*ws
#print abY[1]
#print
#print regression.lwlr(abX[1], abX, abY, 1)




#ridgeWeights = regression.ridgeTest(abX, abY)
#print ridgeWeights

#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(ridgeWeights)
#plt.show()


#xArr, yArr = regression.loadDataSet('abalone.txt')
#regression.stageWise(xArr, yArr, 0.01, 200)
#regression.stageWise(xArr, yArr, 0.001, 5000)


