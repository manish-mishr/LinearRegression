from __future__ import division  # floating point division
import numpy as np
import math
from copy import deepcopy as dc
from copy import copy
import utilities as utils


''' this function reduces the step size with time'''
def diminish_alpha(alpha,count):
    alpha = math.pow(alpha,count)
    return alpha



class Regressor(object):
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest

    def addBias(self,data):
        bias_col = np.ones([len(data), 1])  # None keeps (n, 1) shape
        bias_data = np.append(bias_col, data, 1)
        return bias_data

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest
        
class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params={} ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean
        

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T, Xless)), Xless.T), ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest



''' Batch Regression class and it's methods'''
class BatchRegression(Regressor):
    """
    Batch linear Regression with feature selection
    """
    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5],'alpha': 0.01}
        self.reset(params)
        self.tolerance = math.pow(10, -4)

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]

        train_data = super(BatchRegression, self).addBias(Xless)
        initial = np.random.randn(train_data.shape[1], 1)
        self.weights = utils.normalize(initial, 'l2', axis=0)
        self.batchGradient(train_data, ytrain, numsamples)


    def batchGradient(self, train_data,ytrain,num):
        count = 0
        powerCount = 1
        while True:
            count += 1

            if count%1000 == 0:
                powerCount = int(count/1000)
                self.params['alpha'] = diminish_alpha(self.params['alpha'],powerCount)

            prediction = train_data.dot(self.weights)
            loss = prediction - ytrain.reshape(num,1)
            gradient = train_data.T.dot(loss) / num

            old_weight = dc(self.weights)
            self.weights = self.weights - self.params['alpha'] * gradient
            tolerance = np.linalg.norm(np.subtract(self.weights,old_weight))
            # print tolerance
            if tolerance < self.tolerance:
                break;


    def compute_cost(self,train_data,ytrain,num):
        predictions = train_data.dot(self.weights)
        sqerror = (predictions - ytrain.reshape(num,1))
        cost = sqerror.T.dot(sqerror) * (1 /(2 * num))
        return cost



    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        test_data = super(BatchRegression, self).addBias(Xless)
        ytest = np.dot(test_data, self.weights)
        ytest = np.squeeze(np.asarray(ytest))
        return ytest


''' Ridge Regrssion'''



class RidgeRegression(Regressor):
    """
    Linear Regression with feature selection
    """

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'lambda':0.1}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        Xless = Xtrain[:, self.params['features']]
        length = len(self.params['features'])
        identity = np.identity(length)
        ridge_matrix = self.params['lambda'] * identity
        addRidge = np.add(np.dot(Xless.T,Xless),ridge_matrix)
        self.weights = np.dot(np.dot(np.linalg.inv(addRidge), Xless.T), ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest



'''Stochastic regression '''

class stochasticRegression(Regressor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5],'alpha': 0.01}
        self.reset(params)
        self.epochs = 300

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]

        train_data = super(stochasticRegression, self).addBias(Xless)

        initial = np.random.rand(train_data.shape[1],1)
        self.weights = utils.normalize(initial,'l2', axis=0)
        self.stochasticGradient(train_data,ytrain,numsamples)

    def stochasticGradient(self, train_data,ytrain,num):
        count = 0
        for iter in range(self.epochs):
            count += 1
            if count%50 == 0:
                powerCount = int(count/50)
                self.params['alpha'] = diminish_alpha(self.params['alpha'],powerCount)

            for point in range(len(train_data)):
                prediction = train_data[point,:].dot(self.weights)
                loss = ytrain[point] - prediction
                gradient = train_data[point, :] * loss * (2 / num)
                gradient = gradient.reshape(train_data.shape[1],1)
                old_weight = dc(self.weights)
                self.weights += self.params['alpha'] * gradient
                tolerance = np.linalg.norm(np.subtract(self.weights, old_weight))

    def errorCost(self,train_data,ytrain,num):
        predictions = train_data.dot(self.weights)
        sqerror = (predictions - ytrain.reshape(num, 1))
        cost = sqerror.T.dot(sqerror) * (1 / (2 * num))
        return cost

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        test_data = super(stochasticRegression, self).addBias(Xless)
        ytest = np.dot(test_data, self.weights)
        ytest = np.squeeze(np.asarray(ytest))
        return ytest



'''Lasso Regression'''

class LassoRegression(Regressor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5],'lambda': 0.05}
        self.reset(params)
        self.tolerance = 0.00001

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        train_data = super(LassoRegression, self).addBias(Xless)
        initial = np.random.rand(train_data.shape[1], 1)
        self.weights = utils.normalize(initial, 'l2', axis=0)
        self.lRegression(train_data,ytrain,numsamples)

    def lRegression(self,train_data,ytrain,num):
        tolerance = 10
        while tolerance > self.tolerance:
            for feature in self.params['features']:
                weight = dc(self.weights)
                weight[feature+1,0] = 0

                prediction = train_data.dot(weight)
                residual = np.linalg.norm(np.subtract(prediction,ytrain.reshape(num, 1)))/train_data.shape[1]

                old_weight = dc(self.weights)

                if residual >= -(self.params['lambda']/2) and residual <= self.params['lambda']:
                    self.weights[feature+1,0] = 0
                elif residual > -self.params['lambda']:
                    self.weights[feature+1,0] = residual + (self.params['lambda'] /2)
                else:
                    self.weights[feature + 1, 0] = residual - (self.params['lambda'] / 2)

                tolerance = np.linalg.norm(np.subtract(self.weights, old_weight))


    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        test_data = super(LassoRegression, self).addBias(Xless)
        ytest = np.dot(test_data, self.weights)
        ytest = np.squeeze(np.asarray(ytest))
        return ytest




''' MP Linear Regression'''


class MPLinearRegression(Regressor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5],'epsilon': 1.4}
        self.reset(params)
        self.alpha = 0.001
        self.featureSet = None

    def learn(self, Xtrain, ytrain):

        Xless = Xtrain[:, self.params['features']]
        numsamples = Xtrain.shape[0]
        train_data = super(MPLinearRegression, self).addBias(Xless)
        self.featureSet = self.MPRegression(train_data,ytrain,numsamples)
        print len(self.featureSet)
        Batch = BatchRegression({'features': self.featureSet, 'alpha': 0.01})
        Batch.learn(Xtrain,ytrain)
        self.weights = copy(Batch.weights)

    def MPRegression(self,train_data,ytrain,num):
        initial = np.random.rand(train_data.shape[1], 1)
        self.weights = utils.normalize(initial, 'l2', axis=0)

        residualT = float("inf")
        currentFeature = [0]

        while residualT > self.params['epsilon']:
            bestFeature = None
            corr = float("-inf")
            bestWeight = None


            for feature in list(set(range(len(self.params['features']))) - set(currentFeature)):
                newfeature = copy(currentFeature)
                newfeature.append(feature)

                prediction = train_data[:, newfeature].dot(self.weights[newfeature, :])
                loss = np.subtract(prediction, ytrain.reshape(num, 1))
                gradient = train_data[:,newfeature].T.dot(loss)/num

                weightNew = self.weights[newfeature,:]
                weightNew -= self.alpha * gradient

                predNew = train_data[:, newfeature].dot(weightNew)
                residual = predNew - ytrain.reshape(num,1)

                correlation = train_data[:,newfeature].T.dot(residual)
                pearson = np.linalg.norm(correlation)/num

                if pearson > corr:
                    bestFeature = feature
                    corr = pearson
                    residualT = np.linalg.norm(residual)/num
                    bestWeight = weightNew


            currentFeature.append(bestFeature)
            self.weights[currentFeature,:] = bestWeight

        return currentFeature

    def predict(self, Xtest):
        Xless = Xtest[:, self.featureSet]
        test_data = super(MPLinearRegression, self).addBias(Xless)
        ytest = np.dot(test_data, self.weights)
        ytest = np.squeeze(np.asarray(ytest))
        return ytest