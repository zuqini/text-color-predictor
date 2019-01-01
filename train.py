import logging, sys, json
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# see https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1.0 - sigmoid_x)

class FeedForwardNet:
    def __init__(self, x, y, lr):
        self.x = x
        self.y = y
        self.lr = lr

        self.w1 = np.random.rand(x.shape[1], 4)
        self.b1 = np.zeros((1, 4))
        self.w2 = np.random.rand(4, y.shape[1])
        self.b2 = np.zeros((1, y.shape[1]))

        self.cost = 0

        logging.debug('initialization-------------------------------------')
        logging.debug(self.w1)
        logging.debug(self.w2)

    def feed_forward(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1;
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;
        self.a2 = sigmoid(self.z2)
        logging.debug(self.a2)
        logging.debug('feed forward step-------------------------------------')
        logging.debug(self.x)
        logging.debug(self.w1)
        logging.debug(self.b1)
        logging.debug(self.z1)

    def back_prop(self):
        cost = np.sum(np.square(self.a2 - self.y))
        if abs(cost - self.cost) > 0.01:
            logging.info('cost: ' + str(cost))
        self.cost = cost
        delta_b2 = 2 * (self.y - self.a2) * sigmoid_derivative(self.a2)
        delta_w2 = np.dot(self.a1.T, delta_b2)
        delta_b1 = np.dot(self.w2, delta_b2.T).T * sigmoid_derivative(self.a1)
        delta_w1 = np.dot(self.x.T, delta_b1)

        logging.debug(self.w2)
        logging.debug(self.b2)
        logging.debug(self.w1)
        logging.debug(self.b1)

        self.b2 += self.lr * np.sum(delta_b2, axis=0)
        self.w2 += self.lr * delta_w2
        self.b1 += self.lr * np.sum(delta_b1, axis=0)
        self.w1 += self.lr * delta_w1
        logging.debug('back prop step-------------------------------------')
        logging.debug(sigmoid_derivative(self.a1))
        logging.debug(delta_w2)
        logging.debug(delta_b2)
        logging.debug(delta_w1)
        logging.debug(delta_b1)

    def evaluate(self, X, Y):
        z1 = np.dot(X, self.w1) + self.b1;
        a1 = sigmoid(self.z1)
        z2 = np.dot(self.a1, self.w2) + self.b2;
        a2 = sigmoid(self.z2)

        result = [1 if (prediction == actual).all() else 0 for prediction, actual in zip(np.round(a2), Y)]
        logging.info('result:')
        logging.info(result)
        logging.info('accuracy: ' + str(np.sum(result)/len(Y)))

def gen_input_data(data):
    return list(map(lambda data_point: list(map(lambda rgb: rgb / 255, data_point['backgroundColor'])), data))

def gen_output_data(data):
    return list(map(lambda x: x['textColor'], data))

if __name__ == '__main__':
    with open('data/training-set-v1.json') as f:
        training_set = json.load(f)
    with open('data/test-set-v1.json') as f:
        test_set = json.load(f)

    epoch = 5000
    learning_rate = 0.15
    X = np.array(gen_input_data(training_set))
    Y = np.array(gen_output_data(training_set))
    logging.debug(X.shape)
    logging.debug(Y.shape)

    nn = FeedForwardNet(X, Y, learning_rate)
    for i in range(epoch):
        nn.feed_forward()
        nn.back_prop()

    nn.evaluate(gen_input_data(test_set), gen_output_data(test_set))
