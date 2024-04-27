import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature:list=None, threshold: float=None, left=None,
                    right=None, *, value=None):
        self.feature = feature          # Best feature index to split.
        self.threshold = threshold      # Best threshold to split.
        self.left = left                # The left child node.
        self.right = right              # The right child node.
        self.value = value               # The value of the node if the node is leaf node.

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_sample_split: int=3, max_depth: int=10, n_features: int=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x_train, y_train):
        # Find the best split
        self.n_features = x_train.shape[1] if self.n_features is None \
                            else min(x_train.shape[1], self.n_features)
        self.root = self._grow_tree(x_train, y_train)

    def _grow_tree(self, x, y, depth=0):

        # Create the left and right child nodes
        n_samples, n_feats = x.shape
        n_labels = len(np.unique(y))

        if n_samples <= self.min_sample_split or n_labels == 1 or depth >= self.max_depth:
            return Node(value=self._most_common_label(y))  # Classification problem.

        feature_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thr = self._find_best_split(x, y, feature_idxs)

        # Recursively build the tree
        left_idx, right_idx = self._split(x, best_feature, best_thr)
        left = self._grow_tree(x[left_idx, :], y[left_idx], depth+1)
        right = self._grow_tree(x[right_idx, :], y[right_idx], depth+1)

        # Return the decision node
        return Node(best_feature, best_thr, left=left, right=right)

    def _split(self, x_train, best_feature, best_thr):
        left_idx = np.argwhere(x_train[:, best_feature] < best_thr).flatten()
        right_idx = np.argwhere(x_train[:, best_feature] > best_thr).flatten()
        return left_idx, right_idx

    def _find_best_split(self, x, y, feature_idxs):
        current_entropy = self._calculate_entropy(y)
        current_information_gain = -np.inf
        best_feature = None
        best_thr = None

        for feature_idx in feature_idxs:
            x_feat = x[:, feature_idx]

            for thr in np.unique(x_feat):
                y_left = y[x_feat < thr]
                y_right = y[x_feat > thr]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                information_gain = self._calculate_information_gain(current_entropy,
                                                                    y_left, y_right)

                if information_gain > current_information_gain:
                    current_information_gain = information_gain
                    best_feature = feature_idx
                    best_thr = thr

        return best_feature, best_thr


    def _calculate_information_gain(self, current_entropy, y_left, y_right):
        left_entropy = self._calculate_entropy(y_left)
        right_entropy = self._calculate_entropy(y_right)
        information_gain = current_entropy - (len(y_left)/len(y_right)) * left_entropy \
                            - (len(y_right)/len(y_right)) * right_entropy

        return information_gain

    def _calculate_entropy(self, y):
        y_unique = np.unique(y)
        n = len(y)
        entropy = 0

        for cls in y_unique:
            p = len(y[y == cls]) / n
            entropy += -p * np.log2(p)

        return entropy

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        max_idx = np.argmax(counts)
        max_val = values[max_idx]
        return max_val

    def predict(self, x_test):
        y_pred = []
        for sample in x_test:
            y_pred.append(self._traverse_tree(sample, self.root))

        return np.array(y_pred)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

if __name__ == '__main__':
    data = datasets.load_breast_cancer()

    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(accuracy_score(y_test, predictions))
