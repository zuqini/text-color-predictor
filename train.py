import logging, sys, json
import numpy as np
from result_generator import gen_result_html

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

        self.w1 = np.random.rand(x.shape[1], 8)
        self.b1 = np.zeros((1, 8))
        self.w2 = np.random.rand(8, y.shape[1])
        self.b2 = np.zeros((1, y.shape[1]))

        self.cost = 0

    def feed_forward(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1;
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;
        self.a2 = sigmoid(self.z2)

        cost = np.sum(np.square(self.a2 - self.y))
        if abs(cost - self.cost) > 0.01:
            logging.debug('training cost: ' + str(cost))
        self.cost = cost

    def back_prop(self):
        delta_b2 = 2 * (self.y - self.a2) * sigmoid_derivative(self.a2)
        delta_w2 = np.dot(self.a1.T, delta_b2)
        delta_b1 = np.dot(self.w2, delta_b2.T).T * sigmoid_derivative(self.a1)
        delta_w1 = np.dot(self.x.T, delta_b1)

        self.b2 += self.lr * np.sum(delta_b2, axis=0)
        self.w2 += self.lr * delta_w2
        self.b1 += self.lr * np.sum(delta_b1, axis=0)
        self.w1 += self.lr * delta_w1

    def evaluate(self, X, Y):
        z1 = np.dot(X, self.w1) + self.b1;
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2;
        a2 = sigmoid(z2)
        cost = np.sum(np.square(a2 - Y))
        predictions = np.round(a2)
        result = [1 if (prediction == actual).all() else 0 for prediction, actual in zip(predictions, Y)]
        logging.debug('predictions:')
        logging.debug(list(zip(X, a2)))
        logging.debug('result:')
        logging.debug(result)
        logging.info('accuracy: ' + str(np.sum(result)/len(Y)))
        return [list(zip(X, predictions.tolist())), cost]

def gen_input_data(data):
    return list(map(lambda data_point: list(map(lambda rgb: rgb / 255, data_point['backgroundColor'])), data))

def gen_output_data(data):
    return list(map(lambda x: x['textColor'], data))

if __name__ == '__main__':
    with open('data/data_set_350.json', 'r') as f:
        training_set = json.load(f)
    with open('data/data_set_100.json', 'r') as f:
        validation_set = json.load(f)
    with open('data/data_set_30.json', 'r') as f:
        test_set = json.load(f)

    epoch = 20000
    learning_rate = 0.02
    X = np.array(gen_input_data(training_set))
    Y = np.array(gen_output_data(training_set))

    validation_x = np.array(gen_input_data(validation_set))
    validation_y = np.array(gen_output_data(validation_set))

    nn = FeedForwardNet(X, Y, learning_rate)
    prev_cost = 0
    for i in range(epoch):
        nn.feed_forward()
        nn.back_prop()
        cur_cost = nn.evaluate(validation_x, validation_y)[1]
        if (abs(cur_cost - prev_cost) < 0.2):
            logging.info('early stopping on validation set-----------------------------------')
            logging.info(i)
            break
        logging.info(cur_cost)
        logging.info(prev_cost)
        prev_cost = cur_cost

    logging.info('test set evaluation-----------------------------------')
    result = nn.evaluate(np.array(gen_input_data(test_set)), np.array(gen_output_data(test_set)))[0]
    with open('result.html', 'w') as output:
        output.write(gen_result_html(result))
