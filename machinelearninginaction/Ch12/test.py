import fpGrowth

simDat = fpGrowth.loadSimpDat()
print simDat
print
initSet = fpGrowth.createInitSet(simDat)
print initSet
print
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
myFPtree.disp()
print

freqItems = []
fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)

print freqItems

'''
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
#print parsedDat

initSet = fpGrowth.createInitSet(parsedDat)
print initSet
'''
