import numpy as np
#from scipy import optimize
import matplotlib.pyplot as plt

class trainer(object):
    def __init__(self,N):
        self.N = N
    #def costwrapper(self, params, X, y):
        #self.N.setParams()
        #cost = self.N.cost(X, y)
    #def train(self, X, y):
        
        #params0 = self.N.getParams()
        
        #_res = optimize.minimize(self.costwrapper, params0, jac = True, method='BFGS', )
        

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
        e = sum(.2*(y-self.yp)**2)
        return e
        
    def costp(self, x, y):
        self.yp = self.forward(x)
        
        d3 = np.multiply(-(y-self.yp), self.sigp(self.o2))
        dedM2 = np.dot(self.o2.T, d3)
        
        d2 = np.dot(d3, self.M2.T) *self.sigp(self.o1)
        dedM1 = np.dot(x.T, d2)
        return dedM1, dedM2
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def sigp(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    def learn(self, D1, D2):
        eps = .005
        self.M1 += eps*D1
        self.M2 += eps*D2
    
    def getParams(self):
        params = np.concatenate(self.M1.ravel(), self.M2.ravel())
        return params
    def setParams(self, params):
        M1_start = 0
        M1_end = self.hiddenLayerSize*self.inputLayerSize
        self.M1 = np.reshape(params[M1_start:M1_end], (self.inputLayerSize, self.hiddenLayerSize))
        M2_end = M1_end + self.hiddenLayerSize*self.outputLayerSize
        self.M2 = np.reshape(params[M1_end:M2_end], (self.hiddenLayerSize, self.outputLayerSize))
    def computeGradients(self, X, y):
        dedM1, dedM2 = self.costp(X, y):
            return np.concatenate(dedM1.ravel(),dedM2.ravel())
            
def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    eps = 1e-4
    
    for p in range(len(paramsInitial)):
        perturb[p] = eps
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)
        
        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)
        
        numgrad[p] = (loss2 - loss1) / (2*eps)
        
        perturb[p] = 0
    N.setParams(paramsInitial)
    return numgrad
    
                        
        
def main():
    #input data x
    x = np.array(([1,.5, .7, .6, .2], [.5,.4,.6, .09, .1], [.5,.5,.5,.5,.5]), dtype = float)
    #output data
    y = np.array(([.6],[.33], [.5]), dtype = float)
    nn = Neural()
    costv = []
    costv.extend(nn.cost(x,y))
    for i in range(100):
        dedM1, dedM2 = nn.costp(x,y)
        nn.learn(dedM1, dedM2)
        costv.extend(nn.cost(x,y))
    plt.plot(costv)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()
    print('Matrix 1:')
    print(nn.M1)
    print('Matrix 2:')
    print(nn.M2)

main()