import sys

y = []
for line in open('test'):
    line = line.strip('\n')
    arr = line.split(' ')
    if len(arr) != 7:
        continue
    label = int(arr[0])
    y.append(label)

y_pred = []
for line in open('test.result'):
    line = line.strip('\n')
    arr = line.split('\t')
    if len(arr) != 2:
        continue
    label = int(arr[0])
    y_pred.append(label)

#print len(y)
#print len(y_pred)
#sum = len(y)
sum = 0
num = 0
for i in range(0, len(y)):
    if y[i] != y_pred[i]:
        num += 1
    sum +=1
print num
print sum
print float(num) / float(sum)


