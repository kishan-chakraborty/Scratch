"""
Building a linear regression model from scratch using numpy.

Implement the gradient descent algorithm as well as OLS.
Implement regularization if required. (Ridge and Lasso).
"""

import numpy as np
class LinearRegression:
    """
    Args:
        lr: learning rate for gradient descent
        n_iters: number of iterations
        regularizations: type of regularization (ridge or lasso)
            If not specified, no regularization is used.
        solver: type of solver (gradient descent or OLS)
            OLS is not possible for lasso as it has no exact form.
    """
    def __init__(self,
                 lr: float=0.01,
                 n_iters: int=1000,
                 solver: str = 'OLS',
                 reg: str = None):
        self.lr = lr            # learning rate for gradient descent.
        self.n_iters = n_iters  # number of iterations.
        self.solver = solver    # OLS or Gradient Descent.
        self.reg = reg          # Regularization technique if any.
        self.x = None           # training predictor variables.
        self.y = None           # training response variables.
        self.weights = None     # Calculated regression weight [slope]
        self.bias = None        # Calculated bias [intercept]

    def fit(self, X, y):
        """
        Args:
            X: training predictor variables [n_sample, n_features]
            y: training response variables [n_sample]
        """
        self.X = X
        self.y = y
        if self.solver == 'ols':
            self.ols_solver()
        else:
            self.gradient_descent_solver()

    def ols_solver(self):
        """
        Estimate the regression parameters (weights and biases) using
        Ordinary Least Square method. Least Square Estimates has the least 
        variance among all techniques of calculating these paramters.

        weights (beta_hat) = (X^TX)^-1 X^Ty
        bias (beta_0) = avg(y-X*beta_hat)
        """
        self.weights = np.linalg.inv(self.x.T@self.x) @ (self.x.T @ self.y)
        self.bias = np.avg(self.y-(self.x@self.weights))

    def gradient_descent_solver(self):
        loss = sum((self.y - self.x @ self.weights - self.bias) ** 2)
        for _ in range(self.n_iters):
            pass

    def predict(self, X):
        pass