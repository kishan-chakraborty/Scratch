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
                 solver: str = 'gradient_descent',
                 reg: str = None):
        self.lr = lr            # learning rate for gradient descent.
        self.n_iters = n_iters  # number of iterations.
        self.solver = solver    # OLS or Gradient Descent.
        self.reg = reg          # Regularization technique if any.
        self.x = None           # training predictor variables.
        self.y = None           # training response variables.
        self.n_sample = None    # No. of samples in the training data.
        self.n_features = None  # No. of predictor features.
        self.weights = None     # Calculated regression weight [slope]
        self.bias = None        # Calculated bias [intercept]

    def fit(self, x, y):
        """
        Args:
            x: training predictor variables [n_sample, n_features]
            y: training response variables [n_sample]
        """
        self.x = x
        self.y = y
        self.n_features, self.n_sample = self.x.shape
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
        if self.reg is None:
            self.weights = np.linalg.inv(self.x.T@self.x) @ (self.x.T @ self.y)
            self.bias = np.mean(self.y-(self.x@self.weights))
        else:
            if self.reg == 'ridge':
                self.ridge_solver()
            else:
                print('''Lasso do not have a closed form solution.\n
                      Solving using gradient descent instead.''')
                self.lasso_solver()

    def ridge_solver(self):
        """
                    y = X*beta + beta0 + lambda*(beta^Tbeta)
        Solving the linear regression problem with regularization.
        Regularization becomes necessary to overcome model overfitting.
        Even for a simple model like linear regression over fitting can be a problem
        especially when no. of features is ver high compared to availavle training smaples. 

        The closed form solution is given by:
            beta(λ)=(X^T X + λI)^-1 (X^T y)
            beta0 = avg(y-X*beta)
        """
        if self.solver == 'OLS':
            reg_param = 0.01
            xtx = self.x.T@self.x
            self.weights = np.linalg.inv(xtx + reg_param*np.ones_like(xtx)) @ (self.x.T@self.y)
            self.bias = np.mean(self.y-(self.x@self.weights))
        else:
            self.weights= np.random.rand(self.n_features)
            self.bias= 0

            for _ in range(self.n_iters):
                y_predicted= np.dot(self.x, self.weights)
                self.weights-= (self.lr/ self.n_sample)*(self.x.T).dot(y_predicted- self.y)
                self.bias-= (self.lr/ self.n_sample)*np.sum(y_predicted- self.y)


    def lasso_solver(self):
        pass

    def gradient_descent_solver(self):
        """
        Solve the linear regression problem using gradient descent algorithm.
        This is a implementation of batch gradient descent.
        TO DO: Implement stpchastic or mini batch gradient algorithm
        """
        self.weights= np.random.rand(self.n_features)
        self.bias= 0

        for _ in range(self.n_iters):
            y_predicted= np.dot(self.x, self.weights)
            self.weights-= (self.lr/ self.n_sample)*(self.x.T).dot(y_predicted- self.y)
            self.bias-= (self.lr/ self.n_sample)*np.sum(y_predicted- self.y)

    def predict(self, X):
        pass