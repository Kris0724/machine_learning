from numpy import *
import bayes


listOPosts, listClasses = bayes.loadDataSet()
#print listOPosts, listClasses
#print listClasses

myVocabList = bayes.createVocabList(listOPosts)
#print myVocabList

#retVec = bayes.setOfWords2Vec(myVocabList, listOPosts[0])
#print retVec
#retVec = bayes.setOfWords2Vec(myVocabList, listOPosts[1])
#print retVec

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

#print trainMat

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)

print p0V
print p1V
print pAb


#bayes.testingNB()

#bayes.spamTest()




