import logging, sys
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# see https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
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

        logging.debug('initialization-------------------------------------')
        logging.debug(self.w1)
        logging.debug(self.w2)

    def feed_forward(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1;
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;
        self.a2 = sigmoid(self.z2)
        logging.debug('feed forward step-------------------------------------')
        logging.debug(self.x)
        logging.debug(self.w1)
        logging.debug(self.b1)
        logging.debug(self.z1)

    def back_prop(self):
        delta_b2 = 2 * (self.y - self.a2) * sigmoid_derivative(self.a2)
        delta_w2 = np.dot(self.a1.T, delta_b2)
        delta_b1 = np.dot(self.w2, delta_b2.T).T * sigmoid_derivative(self.a1)
        delta_w1 = np.dot(self.x.T, delta_b1)

        self.b2 += delta_b2
        self.w2 += delta_w2
        self.b1 += delta_b1
        self.w1 += delta_w1
        logging.info('back prop step-------------------------------------')
        logging.debug(sigmoid_derivative(self.a1))
        logging.info(delta_w2)
        logging.info(delta_b2)
        logging.info(delta_w1)
        logging.info(delta_b1)

if __name__ == '__main__':
    X = np.array([[1, 1, 1]])
    Y = np.array([[1, 0]])
    logging.debug(X.shape)
    logging.debug(Y.shape)

    nn = FeedForwardNet(X, Y)
    nn.feed_forward()
    nn.back_prop()
