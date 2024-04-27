import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def calculate_stump_error(weights: np.array,
                          y: np.array,
                          y_pred: np.array) -> float:
    """
    Calculate the error rate of the week classifier m.
    
    Args:
        y: true labels
        y_pred: predicted labels
        weights: Weights assiciated with each sample.

    Returns:
        The classification error.
    """
    num = np.sum(weights * (y != y_pred))
    den = np.sum(weights)

    out = num / den
    return out

def calculate_stump_weight(error: np.array) -> float:
    """
    Calculate the weight associated with the stump based on the error rate.

    Args:
        error: Error rate associated with the stump.
    """
    out = np.log((1 - error) / error)
    return out

def update_weights(weights: np.array,
                   stump_weight: np.array,
                   y: np.array,
                   y_pred: np.array) -> np.array:
    """
    Update weights associated with examples based on mth weak classifier.
    Miss classified examples are weighted more so that the next stump can
    prioritize more.

    Args:
        weights: Current weights assiciated with each example.
        stump_weight: Weight associated with the stump.
        y: True labels.
        y_pred: Predicted labels.

    returns:
        Updated weights associated with each example.
    """
    out = weights * np.exp(stump_weight * (y != y_pred))
    return out

class AdaBoost:
    """
    Implementing the AdaBoost algorithm from scratch.

    Args:
        n_stumps: Number of stumps to use.
        stumps: List of stumps.
        alphas: Weights associated with each stump.
        ws: Weights associated with each example.
        n_sample: Number of training samples.
    """
    def __init__(self,
                 n_stumps: int = 20) -> None:
        self.n_stumps = n_stumps
        self.stumps = []
        self.alphas = np.zeros(n_stumps)
        self.ws = None
        self.n_sample = None

    def fit(self,
            x_train: np.array,
            y_train: np.array) -> None:
        """
        Fit the adaboost model.

        Args:
            x_train: Training data.
            y_train: True labels.
        """
        self.n_sample = x_train.shape[0]

        # Initialize weights to 1/n_sample to each sample.
        self.ws = np.ones(x_train.shape[0]) / self.n_sample

        for i in range(self.n_stumps):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(x_train, y_train, sample_weight=self.ws)
            y_pred = stump.predict(x_train)

            self.stumps.append(stump)

            stump_error = calculate_stump_error(self.ws, y_train, y_pred)

            self.alphas[i] = calculate_stump_weight(stump_error)

            self.ws = update_weights(self.ws, self.alphas[i], y_train, y_pred)

    def predict(self, x_test: np.array) -> np.array:
        """
        Predict the labels for test data.

        Args:
            x_test: Test data.

        Returns:
            Predicted labels
        """
        y_pred = np.zeros(x_test.shape[0])

        for i in range(self.n_stumps):
            y_pred += self.alphas[i] * self.stumps[i].predict(x_test)

        return np.sign(y_pred)

if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    X, y = data.data, data.target
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = AdaBoost(n_stumps=50)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(accuracy_score(y_test, predictions))
