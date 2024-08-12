import torch

REG_FACTOR = 10

class LogisticRegression():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def set_w(self, w):
        self.w = w

    # Get the scores for each class for each training point.
    def get_scores(self, X, w):
        return X @ w.t()
    
class LogisticRegression_w_stochastic_descent(LogisticRegression):
    def __init__(self, X, y):
        super().__init__(X, y)

    def compute_gradient(self):
        # Choose random datapoint
        random_index = torch.randint(0, self.X.shape[0]-1, (1,))
        datapoint = self.X[random_index]
        true_value = self.y[random_index]

        # Calculating loss-function
        pred = torch.log(torch.sum(torch.exp(datapoint @ self.w.t())))
        true_pred = datapoint @ self.w[int(true_value.item())].t()
        loss = (self.w.shape[0] * pred - true_pred) \
                    + REG_FACTOR * torch.sum(self.w ** 2)
        return loss
    
class LogisticRegression_w_standard_descent(LogisticRegression):
    def __init__(self, X, y):
        super().__init__(X, y)

    def compute_gradient(self):
        loss = 0
        for index, datapoint in enumerate(self.X):
            pred = torch.log(torch.sum(torch.exp(datapoint @ self.w.t())))
            true_pred = datapoint @ self.w[int(self.y[index].item())].t()
            loss += (self.w.shape[0] * pred - true_pred) \
                    + REG_FACTOR * torch.sum(self.w ** 2)

        return (1/self.X.shape[0]) * loss