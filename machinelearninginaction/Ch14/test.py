
from numpy import *
#U, Sigma, VT = linalg.svd([[1,7],[7,7]])
#print U
#print 
#print Sigma
#print
#print VT


import svdRec
#Data = svdRec.loadExData()
#U, Sigma, VT = linalg.svd(Data)

#print U
#print
#print Sigma
#print
#print VT

#Sig3=mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
#print U[:,:3]*Sig3*VT[:3,:]

'''
myMat = mat(svdRec.loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2
print myMat
print
print svdRec.recommend(myMat, 2)
#print
#print svdRec.recommend(myMat, 2, simMeas=svdRec.ecludSim)
#print
#print svdRec.recommend(myMat, 2, simMeas=svdRec.pearsSim)
'''


#myMat = mat(svdRec.loadExData2())
myMat = mat(svdRec.loadExData())
print myMat
print svdRec.recommend(myMat, 1, estMethod=svdRec.svdEst)


