import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1.0 - sigmoid_x)

class FeedForwardNet:
    def __init__(self, x, y):
        print('initialization')
        self.x = x
        self.y = y
        self.w1 = np.random.rand(x.shape[1], 4)
        self.b1 = np.zeros((1, 4))
        self.w2 = np.random.rand(4, y.shape[1])
        self.b2 = np.zeros((1, y.shape[1]))
        print(self.w1)
        print(self.w2)

    def feed_forward(self):
        print('feed forward step')
        self.z1 = np.dot(self.x, self.w1) + self.b1;
        print(self.z1)
        self.a1 = sigmoid(self.z1)
        print(self.a1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;
        print(self.z2)
        self.a2 = sigmoid(self.z2)
        print(self.a2)

if __name__ == '__main__':
    X = np.array([[1, 1, 1]])
    print(X.shape)
    Y = np.array([[1, 0]])
    print(Y.shape)

    nn = FeedForwardNet(X, Y)
    nn.feed_forward()
