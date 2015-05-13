import apriori

dataSet = apriori.loadDataSet()

#print dataSet
#print



#C1 = apriori.createC1(dataSet)
#print C1
#print

#D=map(set, dataSet)
#print D
#print
#L1, suppData0 = apriori.scanD(D, C1, 0.5)
#print L1
#print
#print suppData0
#print

#L, suppData = apriori.apriori(dataSet)
#print L
#print 
#print suppData
#print
#print apriori.aprioriGen(L[0], 1)



L,suppData = apriori.apriori(dataSet, minSupport=0.5)
print L
print
print suppData
print

#rules = apriori.generateRules(L, suppData, minConf=0.7)
rules = apriori.generateRules(L, suppData, minConf=0.5)
print rules


#mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
#print mushDatSet[0]
#L,suppData = apriori.apriori(mushDatSet, minSupport=0.3)
#print  L[2]
#for item in L[1]:
#    if item.intersection('2'):print item
