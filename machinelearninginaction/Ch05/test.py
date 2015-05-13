from numpy import *
import logRegres
#dataArr,labelMat = logRegres.loadDataSet()
#weight = logRegres.gradAscent(dataArr, labelMat)
#logRegres.plotBestFit(weight.getA())


#weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
#print weights
#logRegres.plotBestFit(weights)
#logRegres.multiTest()

#weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
#logRegres.plotBestFit(weights)

f_train = open('horseColicTraining.txt')
trainingSet = []
trainingLabels = []
for line in f_train.readlines():
    currLine = line.strip().split('\t')
    lineArr = []
    for i in range(21):
        lineArr.append(float(currLine[i]))
    trainingSet.append(lineArr)
    trainingLabels.append(float(currLine[21]))

weights = logRegres.stocGradAscent1(array(trainingSet), trainingLabels)
#print weights

t_len = len(trainingSet)
yHat = zeros(t_len)
train_err = 0
for i in range(t_len):
    yHat[i] = int(logRegres.classifyVector(array(trainingSet[i]),weights))
    if yHat[i] != int(trainingLabels[i]):
        train_err += 1
print "train_err=",train_err
print "train sum=",t_len
#print yHat

f_test = open('horseColicTest.txt')
testSet = []
testLabels = []
for line in f_test.readlines():
    currLine = line.strip().split('\t')
    lineArr = []
    for i in range(21):
        lineArr.append(float(currLine[i]))
    testSet.append(lineArr)
    testLabels.append(float(currLine[21]))
t_len = len(testSet)
yHat = zeros(t_len)
test_err = 0
for i in range(t_len):
    yHat[i] = int(logRegres.classifyVector(array(testSet[i]),weights))
    if yHat[i] != int(testLabels[i]):
        test_err += 1
print "test_err=",test_err
print "test sum=",t_len
#print yHat



