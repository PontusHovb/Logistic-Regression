import torch
import numpy as np
from matplotlib import pyplot as plt
import time

from LogisticRegression import LogisticRegression_w_stochastic_descent, LogisticRegression_w_standard_descent

THRESHOLD = 100
NUM_ITER = 50
LEARNING_RATE = 0.00001
FILE_DIR = "Data"

"""
Read data from the specified training, validation and test data files.
"""
def read_data(trainFile, valFile, testFile):
    features = []
    labels = []

    # read training, test, and validation data
    for file in [trainFile, valFile, testFile]:
        # read data
        data = np.loadtxt(file)

        # transform into our feature space with "fe()"
        features.append(fe(torch.tensor(data[:,:-1])))

        labels.append(torch.tensor(data[:,-1]))
    
    return features[0], labels[0], features[1], \
        labels[1], features[2], labels[2]

"""
Feature that counts the number of pixels above a specified threshold
in each row and column.
"""
def fe(X):
    # get a "binary image" indicator of pixels above and below the threshold
    X_binary = torch.where(X > THRESHOLD,
        torch.ones_like(X),torch.zeros_like(X)).reshape(-1,28,28)
    
    # calculate row and column features
    X_row = X_binary.sum(dim=1)
    X_col = X_binary.sum(dim=2)

    # feature representing the percentage of pixels with value 0 of total pixels
    X_zeros = torch.where(X == 0,
        torch.zeros_like(X),torch.ones_like(X))
    
    percent_non_zeros = \
        (torch.sum(X_zeros, dim=1)/X_zeros.shape[1]).unsqueeze(1)

    # combine all features
    X_features = torch.cat([X_row, X_col, percent_non_zeros], dim=1)

    return torch.cat([X_features, torch.ones(X_row.shape[0], 
        1, dtype=torch.float64)], dim=1)

"""
Convert "y" into its one-hot-encoding equivalent.
"""
def one_hot(y):
    y_one_hot = torch.zeros([y.shape[0], 10], dtype=torch.float64)
    return y_one_hot.scatter(1, y.reshape(-1, 1).to(torch.long), 1)

def get_scores(X, w):
    return X @ w.t()

"""
Train the model using regularized logistic regression.
"""
def train(X, y, model):
    # convert index labels of y into a one-hot encoding
    one_hot_y = one_hot(y)

    # loss list over iterations for plotting
    losses = []

    # initialize model weights
    w = torch.rand((10, X.shape[1]), dtype=torch.float64, requires_grad=True)
    model.set_w(w)

    i = 0
    while i < NUM_ITER:
        if i % 10 == 0: print(i)
        loss = model.compute_gradient()
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            w.sub_(LEARNING_RATE * w.grad)
        w.grad.data.zero_()
        
        i += 1

    return w, losses

"""
Get list of predicted labels for feature set "X" using model 
parameterized by w.
"""
def predict(X, w):
    # get scores for each class for each input
    scores = get_scores(X, w)

    # find the index of the maximum score for each input,
    # which happens to exactly correspond to the label!
    return torch.argmax(scores, dim=1)

"""
Evaluate the model parameterized by "w", using unseen data features "X" and
corresponding labels "y".
"""
def evaluate(X, y, w):
    # use model to get predictions
    predictions = predict(X, w)
    
    # total number of items in dataset
    total = y.shape[0]

    # number of correctly labeled items in dataset
    correct = torch.sum(predictions == y.long())

    # return fraction of correctly labeled items in dataset
    return float(correct) / float(total)

def main():
    # load data from file
    train_vec, train_lab, val_vec, val_lab, test_vec, test_lab \
        = read_data(f"{FILE_DIR}/train.txt", f"{FILE_DIR}/validate.txt", f"{FILE_DIR}/test.txt")

    # Define model
    model = LogisticRegression_w_standard_descent(train_vec, train_lab)

    # find w through gradient descent
    w, losses = train(train_vec, train_lab, model)

    # evaluate model on validation data
    accuracy = evaluate(val_vec, val_lab, w)

    """
    x.append(reg_factor)
    y.append(accuracy)
    print(reg_factor, ": ",  accuracy)
    """

    print("Validation accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.show()

if __name__ == "__main__":
    main()