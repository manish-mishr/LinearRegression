import numpy as np
from sklearn.preprocessing import normalize

x = np.arange(9).reshape(3,3)
y = np.arange(9).reshape(3,3)

 #recfromcsv('LIAB.ST.csv', delimiter='\t')
# new_col = x.sum(1)[...,None] # None keeps (n, 1) shape
# print new_col
#(210,1)
all_data = np.add(x,y)
print len(all_data)
# all_data *= 2
A= [1,2,3]
B=[1,2,5]

print np.random.rand()
l = 0
for k in range(l+1):
    print l
# print "


mean for  BatchRegression  with alpha  value:  0.9 algorithm is:  0.393246580439
std. dev. for  BatchRegression  with alpha  value:  0.9  algorithm is:  0.416974573333
mean for  BatchRegression  with alpha  value:  0.5 algorithm is:  0.328475455716
std. dev. for  BatchRegression  with alpha  value:  0.5  algorithm is:  0.280880133993
mean for  BatchRegression  with alpha  value:  0.1 algorithm is:  0.306611851275
std. dev. for  BatchRegression  with alpha  value:  0.1  algorithm is:  0.104023756467


Error for StochasticRegression: 0.743003848506
mean for  StochasticRegression  with alpha  value:  0.1 algorithm is:  0.381990040315
std. dev. for  StochasticRegression  with alpha  value:  0.1  algorithm is:  0.1553955346
mean for  StochasticRegression  with alpha  value:  0.05 algorithm is:  0.39522777281
std. dev. for  StochasticRegression  with alpha  value:  0.05  algorithm is:  0.164521677094
mean for  StochasticRegression  with alpha  value:  0.01 algorithm is:  0.419257497831
std. dev. for  StochasticRegression  with alpha  value:  0.01  algorithm is:  0.174671791954



349.056949854
# True
