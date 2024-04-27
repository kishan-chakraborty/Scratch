import numpy as np
from decision_tree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RandomForest(DecisionTree):
    def __init__(self,
                n_trees: int = 10,
                min_sample_split: int = 2,
                max_depth: int = 2,
                n_features: int = None):
        super().__init__()

        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, x_train, y_train):
        for _ in range(self.n_trees):
            tree = DecisionTree(min_sample_split=self.min_sample_split,
                                max_depth=self.max_depth,
                                n_features=self.n_features)

            x_sample, y_sample = self.bootstrap_sample(x_train, y_train)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_sample(self, x, y):
        total_sample = len(x)
        samples = np.random.choice(total_sample, size=total_sample, replace=True)
        return x[samples], y[samples]

    def predict(self, x_test):
        y_pred = np.vstack([tree.predict(x_test) for tree in self.trees]) # [n_trees, test_sample_len]
        out = [self._most_common_label(prediction) for prediction in y_pred.T]
        return np.array(out)

if __name__ == "__main__":
    data = datasets.load_breast_cancer()

    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_features=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(accuracy_score(y_test, predictions))