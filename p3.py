import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

# used in svdecomp f()
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd

class SVDtrain(object):
    def __init__(self, M1, M2):
        self.inputLayer = 5;#number of data points
        self.outputLayer = 1; # output value
        self.hiddenLayer = 3;
        
        # singular - value decomposition 
        self.U1, s1, VT1 = svd(M1.T)
        self.U2, s2, VT2 = svd(M2.T)  

        # create m x n sigma matrix
        sigma1 = zeros((M1.shape[1], M1.shape[0]))
        sigma2 = zeros(( M2.shape[1], M2.shape[0]))
        
        #row1, col1 = np.diag_indices(sigma1.shape[0])
        #row2, col2 = np.diag_indices(sigma2.shape[0])
        
        #sigma1[row1, col1] = s1
        #sigma2[row2, col2] = s2
        
        # populate sigma with n x n diagonal matrix
        sigma1[:M1.shape[1], :M1.shape[1]] = diag(s1)
        sigma2[:M2.shape[1], :M2.shape[1]] = diag(s2)
        
        
        # select top k singular values wanted
        # use f() to look at variance and 
        # opt to remove singular values
        ids1, idv1 = self.sv_threshold(s1, self.inputLayer)
        ids2, idv2 = self.sv_threshold(s2, self.hiddenLayer) 
        
        
        # reconstruct matrix
        
        self.U1 = self.U1[:,[idv1]] 
        sigma1 = sigma1[:,[ids1]]
        VT1 = VT1[[ids1], :]
        
        self.U2 = self.U2[:,[idv2]] 
        sigma2 = sigma2[:, [ids2]]
        VT2 = VT2[[ids2], :]
        

        
        # calculates W1 W2
        self.W1 = np.dot(sigma1, VT1)
        self.W2 = np.dot(sigma2, VT2)
        
        
    
    def sv_threshold(self, s, layercount): 
        id = []
        idv = []
        i = 0
        for singular_value in s:
            if (singular_value >= .02):
                id.append(i) # do not add index in s if sing val is less
                idv.append(i)
                
            i += 1
        if (s.size < layercount):
            i = s.size
            while( i  != layercount):
                id.append(i)
                i += 1
                

        return id, idv
    
    def cost(self, X, y):
        self.zp = self.forward(X)
        e = .5*sum((y-self.zp)**2)
        return e
    
    def costp(self, x, y):
        self.zp = self.forward(x)
        
        d4 = np.multiply(-(y-self.zp), self.sigp(self.p2))
        dedU2 = np.dot(np.dot(self.W2, self.z1), d4)
       
        d3 = np.multiply(d4, self.sigp(self.p2))
        dedW2 = np.dot(self.U2, np.dot(d3.T, self.z1.T))
       
        d2 = np.dot(d3.T, self.sigp(self.p1).T)
        dedU1 = np.dot(np.dot(self.W1, x.T).T, d2.T)
         
        d1 = np.multiply(d2.T,self.sigp(self.p1))
        dedW1 = np.dot(self.U1, np.dot(d1, x))
        
        return dedU2, dedW2, dedU1, dedW1
        
    def forward(self, X):
        
        self.p1 = np.dot(self.U1, np.dot(self.W1, X.T))
        self.z1 = self.sigmoid(self.p1)
        
        temp = np.dot(self.W2, self.z1).T
        self.p2 = np.dot((temp), self.U2.T)
        zp = self.sigmoid(self.p2)
        return zp
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def sigp(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def getParams(self):
        params = np.concatenate((self.U1.ravel(),self.W1.ravel(),self.U2.ravel(),self.W2.ravel()))
        return params
        
    def setParams(self, params):
        
        U1_start = 0
        U1_end = self.hiddenLayer*self.hiddenLayer
        self.U1 = np.reshape(params[U1_start:U1_end], (self.hiddenLayer, self.hiddenLayer))
        
        W1_end = self.hiddenLayer*self.inputLayer 
        self.W1 = np.reshape(params[U1_end:W1_end], (self.hiddenLayer, self.inputLayer))
        
        U2_start = 0
        U2_end = len(self.U2[0])*self.outputLayer
        self.U2 = np.reshape(params[U2_start:U2_end], (self.outputLayer, self.outputLayer))
        
        W2_end = len(self.W2[0])*self.hiddenLayer 
        self.W2 = np.reshape(params[U2_end:W2_end], (self.outputLayer, self.hiddenLayer)) 
        
    def computeGradients(self, X, y):
        dedU2, dedW2, dedU1, dedW1 = self.costp(X, y)
        return np.concatenate((dedU2.ravel(),dedW2.ravel(), dedU1.ravel(),dedW1.ravel()))

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
    emin = sum(np.abs(nn.forward(testx)-testy))/22
    
    for i in range(10):
        nnt = Neural()
        T1 = trainer(nnt)
        T1.train(trainx, trainy)
        plt.plot(T1.e)
        ecur = sum(np.abs(nn.forward(testx)-testy))/22
        if emin > ecur:
            emin = ecur
            M1max = nnt.M1
            M2max = nnt.M2
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
    
def checksvd():
    numdata = pd.read_excel('formatted_vals.xlsx')
    numy = pd.read_excel('yvals.xlsx') 

    trainx = numdata.loc[0:60].to_numpy() 
    testx = numdata.loc[61:82].to_numpy()
    
    trainy = numy.loc[0:60].to_numpy()
    testy =  numy.loc[61:82].to_numpy()

    nn = Neural()
 
    T = trainer(nn)
    T.train(trainx, trainy)
    SVD = SVDtrain(nn.M1, nn.M2)
    Tsvd = trainer(SVD)
    Tsvd.train(trainx, trainy)
    yp = SVD.forward(testx)
    
    plt.plot(Tsvd.e)
    plt.grid(1)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()
    
    print('Predicted :')
    print(yp)
    print('Actual :')
    print(testy)
    
    error = sum(((SVD.forward(testx)-testy)**2)/(testy**2))/22
    print('error:')
    print(error)
    
main()
#checksvd()
#checkgrad()
