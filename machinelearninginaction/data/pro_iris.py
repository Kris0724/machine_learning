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
	f = open('iris.data')
	#f = open('../Ch08/abalone.txt')
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
		
		label = arr[-1]
		if label == 'Iris-setosa':
			label_mat.append(float(1.0))
		if label == 'Iris-versicolor':
			label_mat.append(float(2.0))
		if label == 'Iris-virginica':
			label_mat.append(float(3.0))
		#label_mat.append(float(arr[-1]))		
		line = f.readline()

	#print data_mat
	#print shape(data_mat)
	#print label_mat
	#print shape(label_mat)
	#print data_mat[0:2]
	print regression.standRegres(data_mat, label_mat)

	#print data_mat[0]
	#print label_mat[0]
	#print regression.lwlr(data_mat[0], data_mat, label_mat)
	#return data_mat, label_mat
#readfile1()

#KNN
def readfile2(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())         #get the number of lines in the file
	returnMat = zeros((numberOfLines,4))        #prepare matrix to return
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
		returnMat[index,:] = listFromLine[0:4]
		#print returnMat[index,:]
		#exit(0)
		#classLabelVector.append(int(listFromLine[-1]))
		classLabelVector.append(listFromLine[-1])
 		index += 1
	return returnMat,classLabelVector

def readfile2_2(filename):
	fr = open(filename)
	num_feat = len(open(filename).readline().strip('\t').split(','))
	print "num_feat=",num_feat
	#exit(0)
	numberOfLines = len(fr.readlines())         #get the number of lines in the file
	returnMat = zeros((numberOfLines,num_feat-1))        #prepare matrix to return
	classLabelVector = []                       #prepare labels return   
	fr = open(filename)
	index = 0
	for line in fr.readlines():	
		if line.strip() == '':
			print "line is null"
			continue
		#if line.find('?') == -1:
		#	print "find ?"
		#	continue

		line = line.strip()
		listFromLine = line.split(',')
		returnMat[index,:] = listFromLine[0:num_feat-1]
		classLabelVector.append(listFromLine[-1])
 		index += 1
	return returnMat,classLabelVector

#LOGISTIC REGRES
def readfile3():
	data_mat = []
	label_mat = []
	f = open('./iris.data')
	for line in f.readlines():
		if line.strip() == '':
			print "line is null"
			continue
		line_arr = line.strip().split(',')
		#print line_arr
		#exit(0)	
		data_mat.append([1.0, float(line_arr[0]), float(line_arr[1]), float(line_arr[2]), float(line_arr[3])])
		label = line_arr[-1]
		if label == 'Iris-setosa':
			label_mat.append(float(1.0))
		if label == 'Iris-versicolor':
			label_mat.append(float(2.0))
		if label == 'Iris-virginica':
			label_mat.append(float(3.0))
		#label_mat.append(float(arr[-1]))

	return data_mat, label_mat

#SVM
def readfile4(filename):
	data_mat = []
	label_mat = []
	#f = open('./iris.data.svm')
	f = open(filename)
	for line in f.readlines():
		line_arr = line.strip().split(',')
		data_mat.append([float(line_arr[0]), float(line_arr[1]), float(line_arr[2]), float(line_arr[3])])
		label_mat.append(float(line_arr[4]))
	return data_mat, label_mat

def readfile4_2(filename):
	numfeat = len(open(filename).readline().split(','))
	#print open(filename).readline().split(',')
	print "numfeat=",numfeat
	#exit(0)
	data_mat = []
	label_mat = []
	f = open(filename)
	for line in f.readlines():
		filter = 0
		#print line
		line_arr = []
		cur_arr = line.strip().split(',')
		for i in range(numfeat -1):
			#print "i=",i
			if cur_arr[i] == '?':
				filter =1
				break
			line_arr.append(float(cur_arr[i]))
		if not filter:
			data_mat.append(line_arr)
			label_mat.append(float(cur_arr[-1]))
	return data_mat, label_mat

#ADABOOST
def readfile5():
	numfeat = len(open('./iris.data.svm').readline().split(','))
	print "numfeat=",numfeat
	data_mat = []
	label_mat = []
	f = open('iris.data.svm')
	for line in f.readlines():
		line_arr = []
		cur_arr = line.strip().split(',')
		for i in range(numfeat -1):
			line_arr.append(float(cur_arr[i]))
		data_mat.append(line_arr)
		label_mat.append(float(cur_arr[-1]))
	return data_mat, label_mat

#TREE REGRES
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
#data, label = readfile2('./iris.data')
#data1, label1 = readfile2('./iris.data.svm.train')
#data1, label1 = readfile2_2('./iris.data.svm.train')
data1, label1 = readfile2_2('./wine.data.svm.train')
#data1, label1 = readfile2_2('./breast.svm.train')
#print data1
#print label1
#exit(0)
#data1, label1 = readfile4_2('./iris.data.svm.train')
#print data1
#exit(0)
#data2, label2 = readfile2('./iris.data.svm.test')
#data2, label2 = readfile2_2('./iris.data.svm.test')
data2, label2 = readfile2_2('./wine.data.svm.test')
#data2, label2 = readfile2_2('./breast.svm.test')
#data2, label2 = readfile4_2('./iris.data.svm.test')
#print data
#print label
#inx = [5.0, 3.5, 1.4, 0.2]
normMat1, ranges1, minVals1 = kNN.autoNorm(data1)
#normMat1, ranges1, minVals1 = kNN.autoNorm(mat(data1))
#normMat, ranges, minVals = kNN.autoNorm(data)
normMat2, ranges2, minVals2 = kNN.autoNorm(data2)
#normMat2, ranges2, minVals2 = kNN.autoNorm(mat(data2))
#print normMat1
#exit(0)
m1 = shape(normMat1)[0]
m2 = shape(normMat2)[0]
#errCount = 0
#for i in range(m2):
#    #print kNN.classify0(inx, normMat, label, 3)
#    yHat = kNN.classify0(normMat2[i,:], normMat1, label1, 3)
#    if(yHat != label2[i]):
#	errCount +=1
#print "errCount=",errCount
#print "sum=",m2

#CROSS VALIDATION TO SELECT K
for k in range(20):
	errCount = 0
	for i in  range(m2):
		yHat = kNN.classify0(normMat2[i,:], normMat1, label1, k+1)
		if(yHat != label2[i]):
			errCount +=1
	print "k=",k," errCount=",errCount
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
#data, label = readfile4('./iris.data.svm')
#data, label = readfile4_2('./iris.data.svm.train')
#data, label = readfile4_2('./wine.data.svm.train')
data, label = readfile4_2('./breast.svm.train')
#data_len = len(data)
#print l
#exit(0)
#print data
#print
#print label
#exit(0)
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
#err = 0
#for i in range(data_len):
#	#print i
#	#print datamat[i]
#	y = label[i]
#	y_hat = sign(datamat[i]*mat(ws)+b)
#	if y_hat != y:
#		err = err+1
#print "train err=",err
#print "train sum=",data_len

#data2, label2 = readfile4('./iris.data.svm.test')
#data_len2 = len(data2)
#datamat2 = mat(data2)
#err2 = 0
#for i in range(data_len2):
#	y = label[i]
#	y_hat = sign(datamat2[i]*mat(ws)+b)
#	if y_hat != y:
#		err2 = err2 +1
#print "test err=", err2
#print "test sum=", data_len2
#print datamat[1]
#print label[1]
#print datamat[1]*mat(ws)+b

#OK
#b, alphas = svmMLiA.smoP(data, label, 0.6, 0.001, 40, ('rbf', 1.3))
#OVER FIT
b, alphas = svmMLiA.smoP(data, label, 0.6, 0.001, 40, ('rbf', 0.1))
#LESS FIT
#b, alphas = svmMLiA.smoP(data, label, 0.6, 0.001, 40, ('rbf', 10))
datamat = mat(data)
labelmat = mat(label).transpose()
svind = nonzero(alphas.A>0)[0]
#print "svind=",svind
svs = datamat[svind]
#print "svs=",svs
labelsv = labelmat[svind]
#print "labelsv",labelsv
m,n = shape(datamat)
err_count = 0
for i in range(m):
	#OK
	kernel_eval = svmMLiA.kernelTrans(svs, datamat[i,:], ('rbf', 1.3))
	#OVER FIT
	#kernel_eval = svmMLiA.kernelTrans(svs, datamat[i,:], ('rbf', 0.1))
	#LESS FIT
	#kernel_eval = svmMLiA.kernelTrans(svs, datamat[i,:], ('rbf', 10))
	predict = kernel_eval.T*multiply(labelsv, alphas[svind]) + b
	if sign(predict) != sign(label[i]):
		err_count += 1
print "train err=", err_count
print "train sum=", m

#data2, label2 = readfile4_2('./iris.data.svm.test')
#data2, label2 = readfile4_2('./wine.data.svm.test')
data2, label2 = readfile4_2('./breast.svm.test')
#data_len2 = len(data2)
datamat2 = mat(data2)
lablemat2 = mat(label2)
m,n = shape(datamat2)
err_count = 0
for i in range(m):
	#OK
	kernel_eval = svmMLiA.kernelTrans(svs, datamat2[i,:], ('rbf',1.3))
	#OVER FIT
	#kernel_eval = svmMLiA.kernelTrans(svs, datamat2[i,:], ('rbf',0.1))
	#LESS FIT
	#kernel_eval = svmMLiA.kernelTrans(svs, datamat2[i,:], ('rbf',10))
	predict = kernel_eval.T*multiply(labelsv, alphas[svind]) + b
	if sign(predict) != sign(label2[i]):
		err_count += 1
print "test err=", err_count
print "test sum=", m
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
'''
data = readfile6()
#print data

mymat = mat(data)

#print regTrees.createTree(mymat)
#print regTrees.createTree(mymat, regTrees.modelLeaf, regTrees.modelErr, (1,10))
ws,X,Y = regTrees.linearSolve(mymat)
print ws
'''

