from numpy import *
import svmMLiA

'''
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
print dataArr
print labelArr

#b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
b,alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

print "b=",b
print "alphas=", alphas[alphas > 0]

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print "ws=",ws

datMat = mat(dataArr)
print datMat[0]*mat(ws)+b
print labelArr[0]

print datMat[1]*mat(ws)+b
print labelArr[1]

print datMat[2]*mat(ws)+b
print labelArr[2]

#for i in range(100):
#    if alphas[i] > 0.0:
#        print dataArr[i], labelArr[i]
'''

svmMLiA.testRbf()

#svmMLiA.testDigits(('rbf', 20))



