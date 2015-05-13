from numpy import *
import pca
dataMat = pca.loadDataSet('testSet.txt')
#print dataMat

lowDMat, reconMat = pca.pca(dataMat, 1)
#print shape(lowDMat)
#print lowDMat



