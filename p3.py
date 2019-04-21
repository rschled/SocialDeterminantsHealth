import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

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

        options = {'maxiter': 200, 'disp' : True}
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
        e = .2*sum((y-self.yp)**2)
        return e
        
    def costp(self, x, y):
        self.yp = self.forward(x)
        
        d3 = np.multiply(-(y-self.yp), self.sigp(self.o2))
        dedM2 = np.dot(self.y1.T, d3)
        
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
    
                        
        
def main():
    #input data x

    numdata = pd.read_excel('formatted_vals.xlsx')
    numy = pd.read_excel('yvals.xlsx') 

    trainx = numdata.loc[1:61].to_numpy() 
    testx = numdata.loc[63:83].to_numpy()
    
    trainy = numy.loc[1:61].to_numpy()
    testy =  numy.loc[63:83].to_numpy()

    x = np.array(([1,.5, .7, .6, .2], [.5,.4,.6, .09, .1], [.5,.5,.5,.5,.5]), dtype = float)
    #output data
    y = np.array(([.6],[.33], [.5]), dtype = float)
    nn = Neural()
    
    numgrad = computeNumericalGradient(nn, x, y)
    grad = nn.computeGradients(x, y)
    print(numgrad)
    print(grad)
    
    diff = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
    print('Gradient difference :')
    print(diff)
    
    T = trainer(nn)
    T.train(x, y)
    plt.plot(T.e)
    plt.grid(1)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()
    
    nn.cost(x,y)
    yp = nn.forward(x)
    print('Predicted :')
    print(yp)
    print('Actual :')
    print(y)
    

    
    print('Matrix 1:')
    print(nn.M1)
    print('Matrix 2:')
    print(nn.M2)
    
    #error = sum((nn.forward(xtest)-y)**2)

main()
