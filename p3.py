import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

#class SVDtrain(object):
    #def __init__(self, N):
        #self.N = N
    
    #def cost(self):
    #def costp(self):
    #def forward():
    #def 

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.e.append(self.N.cost(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.cost(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.e = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 1000, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
        

class Neural(object):
    def __init__(self):
        self.inputLayer = 5;#number of data points
        self.outputLayer = 1; # output value
        self.hiddenLayer = 3;
        
        self.M1 = np.random.randn(self.inputLayer, self.hiddenLayer)
        self.M2 = np.random.randn(self.hiddenLayer, self.outputLayer)

    def forward(self, X):
        self.o1 = np.dot(X, self.M1)
        self.y1 = self.sigmoid(self.o1)
        self.o2 = np.dot(self.y1, self.M2)
        yp = self.sigmoid(self.o2)
        return yp
        
    def cost(self,X, y):
        
        self.yp = self.forward(X)
        e = .5*sum((y-self.yp)**2)
        return e
        
    def costp(self, x, y):
        eps = 1e-4
        self.yp = self.forward(x)
        
        d3 = np.multiply(-(y-self.yp), self.sigp(self.o2))
        dedM2 = eps*np.dot(self.y1.T, d3)
        
        d2 = np.dot(d3, self.M2.T) *self.sigp(self.o1)
        dedM1 = eps*np.dot(x.T, d2)
        return dedM1, dedM2
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def sigp(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def getParams(self):
        params = np.concatenate((self.M1.ravel(), self.M2.ravel()))
        return params
        
    def setParams(self, params):
        M1_start = 0
        M1_end = self.hiddenLayer*self.inputLayer
        self.M1 = np.reshape(params[M1_start:M1_end], (self.inputLayer, self.hiddenLayer))
        M2_end = M1_end + self.hiddenLayer*self.outputLayer
        self.M2 = np.reshape(params[M1_end:M2_end], (self.hiddenLayer, self.outputLayer))
    
    def computeGradients(self, X, y):
        dedM1, dedM2 = self.costp(X, y)
        return np.concatenate((dedM1.ravel(),dedM2.ravel()))
            
def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    eps = 1e-4
    
    for p in range(len(paramsInitial)):
        perturb[p] = eps
        N.setParams(paramsInitial + perturb)
        loss2 = N.cost(X, y)
        
        N.setParams(paramsInitial - perturb)
        loss1 = N.cost(X, y)
        
        numgrad[p] = (loss2 - loss1) / (2*eps)
        
        perturb[p] = 0
    N.setParams(paramsInitial)
    return numgrad
    
def checkgrad():
    NN = Neural()
    y = np.array(([.3],[.6],[.9]))
    X = np.array(([.2,.7,.4,.5,.8],[.9,.7,.9,.6,.5],[.5,.7,.8,.1,.3]))
    numgrad = computeNumericalGradient(NN, X, y)
    grad = NN.computeGradients(X,y)
    print(numgrad/grad)                    
        
def main():
    #input data x

    numdata = pd.read_excel('formatted_vals.xlsx')
    numy = pd.read_excel('yvals.xlsx') 

    trainx = numdata.loc[0:60].to_numpy() 
    testx = numdata.loc[61:82].to_numpy()
    
    trainy = numy.loc[0:60].to_numpy()
    testy =  numy.loc[61:82].to_numpy()

    nn = Neural()
 
    T = trainer(nn)
    T.train(trainx, trainy)
    M1max = nn.M1
    M2max = nn.M2
    emin = sum(((nn.forward(testx)-testy)**2)/(testy**2))/22
    
    for i in range(10):
        nnt = Neural()
        T1 = trainer(nnt)
        T1.train(trainx, trainy)
        ecur = sum(((nn.forward(testx)-testy)**2)/(testy**2))/22
        if emin > ecur:
            emin = ecur
            M1max = nnt.M1
            M2max = nnt.M2
    
        
    plt.plot(T.e)
    plt.grid(1)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()
    
    yp = nn.forward(testx)
    print('Predicted :')
    print(yp)
    print('Actual :')
    print(testy)
    

    
    print('Matrix 1:')
    print(M1max)
    print('Matrix 2:')
    print(M2max)
    
    error = sum(((nn.forward(testx)-testy)**2)/(testy**2))/22
    print('error:')
    print(error)

main()
#checkgrad()
