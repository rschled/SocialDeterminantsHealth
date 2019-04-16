import numpy as np

class Neural(object):
    def __init__(self):
        self.inputLayer = 2;#number of data points
        self.outputLayer = 1; # output value
        self.hiddenLayer = 3;
        
        self.M1 = np.random.randn(self.inputLayer, self.hiddenLayer)
        self.M2 = np.random.randn(self.hiddenLayer, self.outputLayer)
    def forward(self, X):
        self.o1 = np.dot(X, self.M1)
        self.s1 = self.sigmoid(self.o1)
        self.o2 = np.dot(self.s1, self.M2)
        self.s2 = self.sigmoid(self.o2)
        return self.s2
        
    def cost(self,y, yp):
        return sum(.5*(y-yp)**2)
    def costp(self, x, y):
        self.yp = self.forward(x)
        d3 = np.multiply(-(y-self.yp), self.sigp(self.s2))
        dSdM2 = np.dot(self.o2.T, d3)
        
        d2 = np.dot(d3, self.M2.T) *self.sigp(self.o1)
        dSdM1 = np.dot(x.T, d2)
        return dSdM1, dSdM2
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def sigp(self, z):
        return -z/(1+np.exp(-z)**2)
        
def main():
    #input data x
    x = np.array(([1,2], [3,4], [5,6]), dtype = float)
    #output data
    y = np.array(([2],[4], [6]), dtype = float)
    nn = Neural()
    yp = nn.forward(x)
    print(yp)

main()