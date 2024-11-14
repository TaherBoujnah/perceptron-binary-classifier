import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split




def train(X, Y):
    global weights, bias
    learning_rate = 0.1
    epochs = 100
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features) * 0.01
    bias = 0
    
    for _ in range(epochs):
        for index, x_i in enumerate(X):
            output = np.dot(x_i, weights) + bias
            y_pred = np.where(output >= 0, 1, 0)
            if y_pred != Y[index]:
                update = learning_rate * (Y[index] - y_pred)
                weights += update * x_i
                bias += update

def infer(X):
    global weights, bias
    output = np.dot(X, weights) + bias
    return np.where(output >= 0, 1, 0)

def generateData(samples=10):
    X, Y = make_classification(n_samples=samples, n_features=10)
    return X, Y

def evaluate():
    X, Y = generateData()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train(X_train, Y_train)
    preds = infer(X_test)
    error_rate = np.mean(preds != Y_test)
    print(f'Error rate on synthetic data: {error_rate:.2f}')

def evaluate_iris():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    mask = Y < 2
    X = X[mask]
    Y = Y[mask]
    train(X, Y)
    predictions = infer(X)
    error_rate = np.mean(predictions != Y)
    print(f'Error rate on Iris dataset: {error_rate:.2f}')

if __name__ == "__main__":
    evaluate()
    evaluate_iris()
