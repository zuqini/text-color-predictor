import logging, sys
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1.0 - sigmoid_x)

class FeedForwardNet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.w1 = np.random.rand(x.shape[1], 4)
        self.b1 = np.zeros((1, 4))
        self.w2 = np.random.rand(4, y.shape[1])
        self.b2 = np.zeros((1, y.shape[1]))

        logging.debug('initialization')
        logging.debug(self.w1)
        logging.debug(self.w2)

    def feed_forward(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1;
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;
        self.a2 = sigmoid(self.z2)
        logging.debug('feed forward step')
        logging.debug(self.z1)
        logging.debug(self.a1)
        logging.debug(self.z2)
        logging.debug(self.a2)

    def back_prop(self):
        pass

if __name__ == '__main__':
    X = np.array([[1, 1, 1]])
    Y = np.array([[1, 0]])
    logging.debug(X.shape)
    logging.debug(Y.shape)

    nn = FeedForwardNet(X, Y)
    nn.feed_forward()
