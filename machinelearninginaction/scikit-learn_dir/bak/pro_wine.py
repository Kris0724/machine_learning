from numpy import *
import sys
sys.path.append('../Ch08/')
import regression

sys.path.append('../Ch02/')
import kNN

sys.path.append('../Ch05/')
import logRegres

sys.path.append('../Ch06/')
import svmMLiA

sys.path.append('../Ch07/')
import adaboost

sys.path.append('../Ch09/')
import regTrees

#STAND REGRES, LWLR
def readfile1():
	f = open('wine.data.all')
	#f = open('../Ch08/abalone.txt')
	#f = open('./iris.data')
	line = f.readline()
	num_feat = len(line.split(',')) - 1
	#num_feat = len(line.split('\t')) - 1
	#print num_feat
	#exit(0)
	data_mat = []
	label_mat = []
	while line:
		if line.strip() == '':
			#print "line is null"
			line = f.readline()
			continue
		line = line.strip('\n')
		#print line
		#arr = line.split('\t')
		arr = line.split(',')
		f_arr = []
		for i in range(num_feat):
			#print arr[i]
			f_arr.append(float(arr[i]))
		#exit(0)
		data_mat.append(f_arr)
		'''
		label = arr[-1]
		if label == 'Iris-setosa':
		label_mat.append(float(1.0))
		if label == 'Iris-versicolor':
			label_mat.append(float(2.0))
		if label == 'Iris-virginica':
			label_mat.append(float(3.0))
		#label_mat.append(float(arr[-1]))
		'''
		label_mat.append(float(arr[-1]))
		line = f.readline()

	#print data_mat
	#print shape(data_mat)
	#print label_mat
	#print shape(label_mat)
	#print data_mat[0:2]
	#print regression.standRegres(data_mat, label_mat)

	#print data_mat[0]
	#print label_mat[0]
	#print regression.lwlr(data_mat[0], data_mat, label_mat)
	return data_mat, label_mat
'''
#readfile1()
data, label = readfile1()
#print data
#print label
#print shape(data)
#print shape(label)
#print regression.standRegres(data, label)
print data[1]
print label[1]
print regression.lwlr(data[1], data, label)
'''

def readfile2(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())         #get the number of lines in the file
	returnMat = zeros((numberOfLines,13))        #prepare matrix to return
	classLabelVector = []                       #prepare labels return   
	fr = open(filename)
	index = 0
	for line in fr.readlines():	
		if line.strip() == '':
			print "line is null"
			continue
		line = line.strip()
		listFromLine = line.split(',')
		#print listFromLine
		#exit(0)
		#print listFromLine[0:4]
		#exit(0)
		returnMat[index,:] = listFromLine[0:13]
		#print returnMat[index,:]
		#exit(0)
		#classLabelVector.append(int(listFromLine[-1]))
		classLabelVector.append(listFromLine[-1])
 		index += 1
	return returnMat,classLabelVector

def readfile3():
	data_mat = []
	label_mat = []
	#f = open('./iris.data')
	f = open('./wine.data.all')
	for line in f.readlines():
		if line.strip() == '':
			print "line is null"
			continue
		line_arr = line.strip().split(',')
		#print line_arr
		#exit(0)	
		data_mat.append([1.0, float(line_arr[0]), float(line_arr[1]), float(line_arr[2]), float(line_arr[3]), float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]), float(line_arr[8]), float(line_arr[9]), float(line_arr[10]), float(line_arr[11]), float(line_arr[12])])
		#label = line_arr[-1]
		#if label == 'Iris-setosa':
		#	label_mat.append(float(1.0))
		#if label == 'Iris-versicolor':
		#	label_mat.append(float(2.0))
		#if label == 'Iris-virginica':
		#	label_mat.append(float(3.0))
		label_mat.append(float(line_arr[-1]))

	return data_mat, label_mat

def readfile4():
	data_mat = []
	label_mat = []
	f = open('./wine.data.all.svm')
	for line in f.readlines():
		line_arr = line.strip().split(',')
		#data_mat.append([float(line_arr[0]), float(line_arr[1]), float(line_arr[2]), float(line_arr[3])])
		data_mat.append([float(line_arr[0]), float(line_arr[1]), float(line_arr[2]), float(line_arr[3]), float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]), float(line_arr[8]), float(line_arr[9]), float(line_arr[10]), float(line_arr[11]), float(line_arr[12])])
		label_mat.append(float(line_arr[13]))
	return data_mat, label_mat

def readfile5():
	numfeat = len(open('./wine.data.all.svm').readline().split(','))
	print "numfeat=",numfeat
	data_mat = []
	label_mat = []
	f = open('wine.data.all.svm')
	for line in f.readlines():
		line_arr = []
		cur_arr = line.strip().split(',')
		for i in range(numfeat -1):
			line_arr.append(float(cur_arr[i]))
		data_mat.append(line_arr)
		label_mat.append(float(cur_arr[-1]))
	return data_mat, label_mat

def readfile6():
	data_mat = []
	f = open('iris.data.svm')
	for line in f.readlines():
		cur_line = line.strip().split(',')
		flt_line = map(float, cur_line)
		data_mat.append(flt_line)
	return data_mat


#KNN
'''
data, label = readfile2('./wine.data.all')
#print data
#print label
inx = [5.0, 3.5, 1.4, 0.2,1,1,1,1,1,1,1,1,1]
normMat, ranges, minVals = kNN.autoNorm(data)
print kNN.classify0(inx, normMat, label, 3)
'''

#LOGISTIC REGRES
'''
data, label = readfile3()
#print data
#print label
#weights = logRegres.gradAscent(data, label)
weights = logRegres.stocGradAscent1(array(data), label)
print weights
'''

#SVM
'''
data, label = readfile4()
#print data
#print label
#b, alphas = svmMLiA.smoSimple(data, label, 0.6, 0.001, 40)
#print b,alphas
#print b
#print alphas[alphas > 0]

#b, alphas = svmMLiA.smoP(data, label, 0.6, 0.001, 40)
#print b
#print alphas[alphas > 0]
#ws = svmMLiA.calcWs(alphas, data, label)
#print ws
#datamat = mat(data)
#print datamat[0]
#print label[0]
#print datamat[0]*mat(ws)+b

b, alphas = svmMLiA.smoP(data, label, 0.6, 0.001, 40, ('rbf', 1.3))
datamat = mat(data)
labelmat = mat(label)
svind = nonzero(alphas.A>0)[0]
print svind
svs = datamat[svind]
print svs
'''

#ADABOOST
'''
data, label = readfile5()
#print data
#print label
weak, est = adaboost.adaBoostTrainDS(data, label, 9)
print weak
#print est
'''

#TREE REGRES

data = readfile6()
#print data
mymat = mat(data)
print regTrees.createTree(mymat)
#print regTrees.createTree(mymat, regTrees.modelLeaf, regTrees.modelErr, (1,10))
#ws,X,Y = regTrees.linearSolve(mymat)
#print ws


