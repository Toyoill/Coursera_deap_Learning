import numpy as np
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_y, test_set_y_orig, classes = load_dataset()


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def initializae_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0.
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    loss = -(Y.T * np.log(A) + (1 - Y.T) * np.log(1 - A))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    cost = np.sum(loss) / m
    return dw, db, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.5):
    for _ in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b


def prediction(w, b, X):
    ret = np.zeros((1, X.shape[1]))
    result = sigmoid(np.dot(w.T, X) + b)
    for i in range(result.shape[1]):
        ret[0, i] = 1. if result[0, i] > 0.5 else 0.
    return ret


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    print("start")

    w, b = initializae_parameters(X_train.shape[0])
    w, b = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    predict = prediction(w, b, X_test)
    
    return predict
