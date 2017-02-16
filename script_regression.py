from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs
import utilities as util
from collections import  defaultdict
from sklearn.cross_validation import ShuffleSplit
import time
# import plotfcns

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

def multipleSplits(total,splitDic):
    #(25 % train data 75% test data)
    partitions = [0.2, 0.4, 0.6, 0.8]

    count = 0
    for num in partitions:
        trainsize = int(num * total)
        testsize = int((1-num) * total)
        splitDic[count] = (trainsize,testsize)
        count += 1

def bootstrap():
    dataindex = ShuffleSplit(n=total_data, n_iter=4, test_size=0.25, random_state=10)
    return dataindex

if __name__ == '__main__':
    start = time.time()
    total_data = 10000

    numparams = 3
    numruns = 5
    
    regressionalgs = {#'Random': algs.Regressor(),
                 # 'Mean': algs.MeanPredictor(),
    #             'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
    #             'FSLinearRegression50': algs.FSLinearRegression({'features': range(60)}),
    #             'RidgeRegression': algs.RidgeRegression({'features': range(350)}),
                'BatchRegression': algs.BatchRegression({'features': range(385)}),
    #             'StochasticRegression': algs.stochasticRegression({'features': range(385)})
    #             'LassoRegression': algs.LassoRegression({'features': range(385)}),
    #             'MPLinearRegression': algs.MPLinearRegression({'features': range(385)})

    }
    numalgs = len(regressionalgs)

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))



    splitDic = defaultdict(tuple)

    # functions calls and split data for various partitions
    multipleSplits(total_data,splitDic)


    ''' iterate over each split and calculate error'''




    parameters = {'RidgeRegression': {'lambda': [0.01, 0.1, 1]},
                  'Random': {},
                  'Mean': {},
                  'FSLinearRegression5': {},
                  'FSLinearRegression50': {},
                  'BatchRegression': {'alpha': [0.1, 0.05, 0.01]},
                  'StochasticRegression': {'alpha': [0.1, 0.05, 0.01]},
                  'LassoRegression': {'lambda': [0.09, 0.05, 0.01]},
                  'MPLinearRegression': {'epsilon': [1.6, 1.5, 1.4]}
                  }




    for learnername, learner in regressionalgs.iteritems():
        ''' Select the parameter for algorithm'''
        items = parameters[learnername]

        run = 0
        ''' Create train data and test data for various split and normalize them'''
        for key, value in splitDic.iteritems():
            trainsize, testsize = value

            trainset, testset = dtl.load_ctscan(trainsize, testsize)

            shuffleIndex = bootstrap()
            for train,test in shuffleIndex:
                trainData = train
                testData = test



            trainnorm = util.normalize(trainset[0], 'l2', axis=0)
            testnorm = util.normalize(testset[0], 'l2', axis=0)

            print('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0])


            if len(items) > 0:
                key = items.keys()
                values = items.values()
                numParams = len(values[0])
            else:
                numParams = 1

            for param in range(numParams):
                '''select each parameter once and run the program'''
                if numParams == 1:
                    params ={}
                else:
                    params = {key[0]: values[0][param]}
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params)
                print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
                # Train model
                learner.learn(trainnorm, trainset[1])
                # Test model
                predictions = learner.predict(testnorm)
                error = geterror(predictions,testset[1])
                print 'Error for ' + learnername + ': ' + str(error)
                errors[learnername][param,run] = error

            run += 1


    ''' Print mean and std. dev for each learner algorithm'''


    for key,value in errors.iteritems():


         items = parameters[key]

         if len(items) > 0:
             key1 = items.keys()
             values = items.values()
             numParams = len(values[0])
         else:
             numParams = 1

         if numParams > 1:
             for num in range(numParams):
                 print "mean for  " + str(key) + "  with "+ str(key1[0]) +"  value:  " +str(values[0][num])+  " algorithm is:  " + str(np.mean(value[num,:]))
                 print "std. dev. for  " + str(key) + "  with "+ str(key1[0]) +"  value:  " +str(values[0][num])+ "  algorithm is:  " + str(np.std(value[num,:]))
         else:
             print "mean for " +str(key) +" algorithm is:  " + str(np.mean(value[0,:]))
             print "std. dev. for  " + str(key) +"  algorithm is:  " + str(np.std(value[0,:]))

         end = time.time()
         print "total time taken: ",str(end-start)

