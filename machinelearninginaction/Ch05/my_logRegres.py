'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
#import np
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    #fr = open('testSet.txt')
    fr = open('tt.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        dataMat.append([1.0, float(lineArr[0])])
        labelMat.append(float(lineArr[1]))
    #print dataMat
    #print labelMat
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    #print dataMatrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrixa
    #print labelMat
    m,n = shape(dataMatrix)
    #print m,n
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    #print weights
    #exit(0)
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        #print dataMatrix*weights
        #print h
        #print dataMatrix
        #print weights
        #exit(0)
        error = (labelMat - h)              #vector subtraction
        #print error
        #exit(0)
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
        #print weights
        #exit(0)
    return weights

def gradDescent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    #print dataMatrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrixa
    #print labelMat
    m,n = shape(dataMatrix)
    #print m,n
    alpha = 0.001
    maxCycles = 5000
    weights = ones((n,1))
    #print weights
    #exit(0)
    cost = []
    for k in range(maxCycles):              #heavy on matrix operations
        h = dataMatrix*weights     #matrix mult
        #print dataMatrix*weights
        #print h
        #print dataMatrix
        #print weights
        #exit(0)
        error = (h - labelMat)              #vector subtraction
        #error = (labelMat - h)              #vector subtraction
        cost.append(np.sum(array(error)**2))
        #print error
        #exit(0)
        weights = weights - alpha * dataMatrix.transpose()* error #matrix mult
        #print weights
        #exit(0)
    #print cost
    cx = range(len(cost))
    plt.figure(1)
    plt.plot(cx,cost)
    plt.ylim(0,5)
    plt.figure(2)
    plt.plot(dataMatrix[:,1],labelMat,'b.')
    x = np.arange(0,6,0.1)
    w = array(weights)
    #print w
    #exit(0)
    y = x * w[1] + w[0]
    plt.plot(x,y)
    plt.margins(0.2)
    plt.show()


    return weights

dat,lab = loadDataSet()
#print dat,lab
w = gradDescent(dat, lab)
#print w

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    #print m,n
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    #print weights
    for i in range(m):
	#print sum(dataMatrix[i]*weights)
        h = sigmoid(sum(dataMatrix[i]*weights))
        #print h
        error = classLabels[i] - h
        #print error
        #exit(0)
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    #print len(trainWeights)
    #exit(0)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        
