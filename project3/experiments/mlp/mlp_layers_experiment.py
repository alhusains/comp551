import pickle
import numpy as np
import matplotlib.pyplot as plt


#retirive data from CIFAR-10 dataset
def unpickle(fileName):
    with open(fileName, 'rb') as f:
        dict = pickle.load(f, encoding= "bytes")
    return dict

#merge the bactches of CIFAR, as we have 1 to 5 
#load_num represents the number of batches to load
def merge_batches(num_to_load=1):
    for i in range(1):
        fileName = "data/cifar-10-batches-py/data_batch_" + str(i + 1)
        data = unpickle(fileName)
        if i == 0:
            features = data[b'data']
            labels = np.array(data[b'labels'])
        else:
            features = np.append(features, data["data"], axis=0)
            labels = np.append(labels, data["labels"], axis=0)
    return features, labels

#one-hot-encode the target label
def one_hot_encode(data):
    one_hot = np.zeros((data.shape[0], 10))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot

#Normalizing the Pixel Values, input is the list of image pixel values
def normalize(data):
    return data / 255.0

#helper function for the pre_processing, input is the number of batches to load
def preprocess(num_to_load=1):
    X, y = merge_batches(num_to_load=1)
    X = normalize(X)
    X = X.reshape(-1, 3072, 1)
    y = one_hot_encode(y)
    y = y.reshape(-1, 10, 1)
    return X, y

#splitting the data into training and validation
def dataset_split(X, y, ratio=0.8):
    split = int(ratio * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    training_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[training_idx, :], X[val_idx, :]
    y_train, y_val = y[training_idx, :], y[val_idx, :]
    print("Records in Training Dataset", X_train.shape[0])
    print("Records in Validation Dataset", X_val.shape[0])
    return X_train, y_train, X_val, y_val

#sigmoid activation
def sigmoid(out):
    return 1.0 / (1.0 + np.exp(-out))

#sigmoid derivative
def delta_sigmoid(out):
    return sigmoid(out) * (1 - sigmoid(out))
#sigmoid cross entropy
def SigmoidCrossEntropyLoss(a, y):
    return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

#the class to define our MLP structure 
class MLP(object):
    #initialize the biases and weights using a Gaussian distribution with mean 0, and variance 1.
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        #setting appropriate dimensions for weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    #passing image features to the MLP
    def feedforward(self, x):
        activation = x
        activations = [x]  # list to store activations for every layer
        outs = []  # list to store out vectors for every layer
        for b, w in zip(self.biases, self.weights):
            out = np.dot(w, activation) + b
            outs.append(out)
            activation = sigmoid(out)
            activations.append(activation)
        return outs, activations

    #Data iter to for batching
    def get_batch(self, X, y, batch_size):
        for batch_idx in range(0, X.shape[0], batch_size):
            batch = zip(X[batch_idx:batch_idx + batch_size],
                        y[batch_idx:batch_idx + batch_size])
            yield batch
    
    #training phase
    def train(self, X, y, X_val, y_val, batch_size=100, learning_rate=0.2, epochs=1000):
        n_batches = int(X.shape[0] / batch_size)
        acc_array = []
        for j in range(epochs):
            batch_iter = self.get_batch(X, y, batch_size)
            for i in range(n_batches):
                batch = next(batch_iter)
                # same shape as self.biases
                del_b = [np.zeros(b.shape) for b in self.biases]
                # same shape as self.weights
                del_w = [np.zeros(w.shape) for w in self.weights]
                for batch_X, batch_y in batch:
                    # accumulate all the bias and weight gradients
                    loss, delta_del_b, delta_del_w = self.backpropagate(
                        batch_X, batch_y)
                    del_b = [db + ddb for db, ddb in zip(del_b, delta_del_b)]
                    del_w = [dw + ddw for dw, ddw in zip(del_w, delta_del_w)]
            accuracy = self.eval(X_val, y_val)
            self.weights = [w - (learning_rate / batch_size)
                            * delw for w, delw in zip(self.weights, del_w)]
            self.biases = [b - (learning_rate / batch_size)
                           * delb for b, delb in zip(self.biases, del_b)]
            print("\nEpoch %d complete\tLoss: %f\n" % (j, loss))
            acc_array.append(accuracy)
        return acc_array

    def backpropagate(self, x, y):
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        outs, activations = self.feedforward(x)
        loss = SigmoidCrossEntropyLoss(activations[-1], y) #cost function
        # calculate derivative of cost Sigmoid Cross entropy which is to be minimized
        delta_cost = activations[-1] - y
        # backward pass to reduce cost
        # gradients at output layers
        delta = delta_cost
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].T)

        # updating gradients of each layer using reverse or negative indexing
        for l in range(2, self.num_layers):
            out = outs[-l]
            delta_activation = delta_sigmoid(out)
            delta = np.dot(self.weights[-l + 1].T, delta) * delta_activation
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l - 1].T)
        return (loss, del_b, del_w)

    #Evaluation Phase
    def eval(self, X, y):
        count = 0
        for x, _y in zip(X, y):
            outs, activations = self.feedforward(x)
            # postion of maximum value is the predicted label
            if np.argmax(activations[-1]) == np.argmax(_y):
                count += 1
        print("Accuracy: %f" % ((float(count) / X.shape[0]) * 100))
        return ((float(count) / X.shape[0]) * 100)

    def predict(self, X):
        labels = unpickle("data/cifar-10-batches-py/batches.meta")[b"label_names"]
        preds = np.array([])
        for x in X:
            outs, activations = self.feedforward(x)
            preds = np.append(preds, np.argmax(activations[-1]))
        preds = np.array([labels[int(p)] for p in preds])
        return preds

"""Constructing the first MLP with 3 layers with the function below"""

def three_layer():
    X, y = preprocess(num_to_load=1)
    X_train, y_train, X_val, y_val = dataset_split(X, y)
    model = DNN([3072, 75, 10])  # initialize the model
    acc_array = model.train(X_train, y_train, X_val, y_val, epochs=100)  # train the model
    model.eval(X_val, y_val)  # check accuracy using validation set
    # preprocess test dataset
    test_X = unpickle("data/cifar-10-batches-py/test_batch")[b'data'] / 255.0
    test_X = test_X.reshape(-1, 3072, 1)
    # make predictions of test dataset
    print(model.predict(test_X))
    return acc_array


#the accuracy array
acc_array = three_layer()

"""Constructing the MLP with 5 layers"""

def five_layer():
    X, y = preprocess(num_to_load=1)
    X_train, y_train, X_val, y_val = dataset_split(X, y)
    model = DNN([3072, 1000, 100, 50, 10])  # initialize the model
    acc_array2 = model.train(X_train, y_train, X_val, y_val, epochs=100)  # train the model
    model.eval(X_val, y_val)  # check accuracy using validation set
    # preprocess test dataset
    test_X = unpickle("data/cifar-10-batches-py/test_batch")[b'data'] / 255.0
    test_X = test_X.reshape(-1, 3072, 1)
    # make predictions of test dataset
    print(model.predict(test_X))
    return acc_array2

#accuracy array
acc_array2 = five_layer()

"""Plotting the accuracy graph for the two MLP structures defined above"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(acc_array, 'r', label='With 3 layers')
plt.plot(acc_array2, 'g', label='With 5 layers')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Experimenting with Number of layers')
plt.legend()
plt.show()