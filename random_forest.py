import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier


class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=max_depth, criterion="entropy")
            for i in range(n_estimators)
        ]
        self.alphas = (
            []
        )  # Importance values for adaptive boosting extra credit implementation

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        TODO:
        - Set random seed if it is inputted
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Args:
        - num_training: number of data points in the bootstrapped dataset.
        - num_features: number of features in the original dataset.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        Hint 1: Please use np.random.choice. First get the row_idx first, and then second get the col_idx.
        Hint 2:  If you are getting a Test Failed: 'bool' object has no attribute 'any' error, please try flooring, or converting to an int, the number of columns needed for col_idx. Using np.ceil() can cause an autograder error.
        """
        np.random.seed(seed = random_seed)
        row = np.random.choice(num_training,num_training)
        size = int(num_features*self.max_features)
        col = np.random.choice(num_features, size, replace=False)
        return row, col

    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        np.random.seed(self.random_seed)
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        X: NxD numpy array, where N is number
           of instances and D is the dimensionality of each
           instance
        y: 1D numpy array of size (N,), the predicted labels
        Returns:
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        self.bootstrapping(X.shape[0], X.shape[1])
    
        for i in range(self.n_estimators):
            bootstrap_X = X[self.bootstraps_row_indices[i]][:, self.feature_indices[i]]
            bootstrap_y = y[self.bootstraps_row_indices[i]]
            self.decision_trees[i].fit(bootstrap_X, bootstrap_y)

    def adaboost(self, X, y):
        """
        TODO:
        - Implement AdaBoost training by adjusting sample weights after each round of training a weak learner.
        - Begin by initializing equal weights for each training sample.
        - For each weak learner:
            - Train the learner on the current sample weights.
            - Get predictions for the training data using the learner's `predict()` method.
            - Calculate the weighted error of the sample and normalize.
            - Calculate `alpha`, the importance of the tree, using the formula from the notebook, and store it.
            - Update the weights of the samples using the formula from the notebook and normalize.

        Args:
        - X: NxD numpy array, feature set.
        - y: 1D numpy array of size (N,), labels.
        Returns:
            None. Trains the ensemble using AdaBoost's weighting mechanism.
        """
        n_samples, n_features = X.shape
        sample_weights = np.ones(n_samples) / n_samples

        y_binary = np.where(y == 1, 1, -1)

        for i in range(self.n_estimators):
            # Train weak learner
            self.decision_trees[i].fit(X, y_binary, sample_weight=sample_weights)
            
            # Get predictions
            predictions = self.decision_trees[i].predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y_binary
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Avoid division by zero
            error = np.clip(error, 1e-15, 1 - 1e-15)
            
            # Calculate alpha (importance of the tree)
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y_binary * predictions)
            sample_weights /= np.sum(sample_weights)
            
            # Early stopping if error is very low
            if error < 0.01:
                break

        # Normalize alphas
        self.alphas = np.array(self.alphas)
        self.alphas /= np.sum(self.alphas)
    def OOB_score(self, X, y):
        # Helper function. You don't have to modify it.
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def predict_adaboost(self, X):
        # Helper method. You don't have to modify it.
        # This function makes predictions using AdaBoost ensemble by aggregating weighted votes.
        N = X.shape[0]
        weighted_votes = np.zeros((N, 7))

        for alpha, tree in zip(self.alphas, self.decision_trees[: len(self.alphas)]):
            pred = tree.predict(X)
            for i in range(N):
                class_index = int(pred[i])
                weighted_votes[i, class_index] += alpha

        return np.argmax(weighted_votes, axis=1)

    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in
        one decision tree of your choice from the tuned random_forest from Q3.2.
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        importances = self.decision_trees[0].feature_importances_

        # Sort feature importances and filter out zero importances
        indices = np.argsort(importances)
        sorted_importances = importances[indices]
        
        # Get feature names using indices
        feature_names = data_train.columns[self.feature_indices[0]][indices]
        
        # Filter non-zero importances
        non_zero_mask = sorted_importances > 0
        sorted_importances = sorted_importances[non_zero_mask]
        sorted_feature_names = feature_names[non_zero_mask]

        # Create horizontal bar plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(sorted_importances))
        plt.barh(y_pos, sorted_importances)
        plt.yticks(y_pos, sorted_feature_names)
        plt.title("Feature Importance for Decision Tree")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    def select_hyperparameters(self):
        """
        Hyperparameter tuning Question
        TODO: assign a value to n_estimators, max_depth, max_features
        Args:
            None
        Returns:
            n_estimators: int number (e.g 2)
            max_depth: int number (e.g 4)
            max_features: a float between 0.0-1.0 (e.g 0.1)
        """
        n_estimators = 3
        max_depth = 15
        max_features = 1.0
        return n_estimators, max_depth, max_features
        raise NotImplementedError()
