import pandas as pd
import numpy as np


def preprocess_classification_dataset():
    # train dataset
    train_df = pd.read_csv("train.csv")
    # grab all columns except the last one
    train_feat_df = train_df.iloc[:, :-1]
    train_output = train_df[['output']]
    X_train = train_feat_df.values
    y_train = train_output.values
    # val dataset
    val_df = pd.read_csv("val.csv")
    val_feat_df = val_df.iloc[:, :-1]  # grab all columns except the last one
    val_output = val_df[['output']]
    X_val = val_feat_df.values
    y_val = val_output.values
    # test dataset
    test_df = pd.read_csv("test.csv")
    test_feat_df = test_df.iloc[:, :-1]  # grab all columns except the last one
    test_output = test_df[['output']]
    X_test = test_feat_df.values
    y_test = test_output.values

    return X_train, y_train, X_val, y_val, X_test, y_test


# k-nearest neighbors.
def knn_classification(X_train, y_train, x_new, k=5):
    euclidean_distance = (np.sum((X_train - x_new) ** 2, axis=1))**0.5
    if k == 1:
        nn = np.argmin(euclidean_distance, axis=0)
        y_new_pred = y_train[nn][0]
    else:
        knn = np.argsort(euclidean_distance)[:k]
        y_knn, count = np.unique(y_train[knn], return_counts=True)
        y_new_pred = y_knn[np.argmax(count)]
    return y_new_pred


def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    ref = X_train
    ones = np.ones((len(ref), 1), dtype=int)
    X = np.hstack((ones, ref))
    XT = np.transpose(X)
    num_of_features = X.shape[1]
    np.random.seed(random_seed)  # for reproducibility
    weights = np.random.normal(loc=0.0, scale=1.0, size=(num_of_features, 1))
    for i in range(max_iters):
        weights = weights - (alpha) * \
            (XT @ (sigmoid_function(X @ weights) - y_train))
    return weights


def sigmoid_function(z):
    result = 1.0/(1.0 + np.exp(-z))
    return result


def logistic_regression_prediction(X, weights, threshold=0.5):
    ref = X
    ones = np.ones((len(ref), 1), dtype=int)
    X_ones = np.hstack((ones, ref))
    y_preds = []
    for i in range(len(X_ones)):
        Y = sigmoid_function(X_ones @ weights)
        if (Y[i] >= threshold):
            y_preds.append([1])
        else:
            y_preds.append([0])
    return np.array(y_preds)

# helper function for knn computions for an entire dataset.
def knn_computions(X_train, y_train, X, k):
    predictions = []
    for i in range(len(X)):
        predictions.append(knn_classification(X_train, y_train, X[i], k))
    return np.array(predictions)

# helper function for test accuracy computions.
def test_accuracy_computions(X_merge, y_merge, X_test, y_test, best_method, alpha, max_iters, random_seed, threshold):
    if best_method == '1nn':
        prediction = knn_computions(X_merge, y_merge, X_test, 1)
    if best_method == '3nn':
        prediction = knn_computions(X_merge, y_merge, X_test, 3)
    if best_method == '5nn':
        prediction = knn_computions(X_merge, y_merge, X_test, 5)
    if best_method == 'logistic regression':
        trained_weights = logistic_regression_training(
            X_merge, y_merge, alpha, max_iters, random_seed)
        prediction = logistic_regression_prediction(
            X_test, trained_weights, threshold)
    test_accuracy = (y_test.flatten() == prediction.flatten()
                     ).sum() / y_test.shape[0]
    return test_accuracy


def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()
    # knn prediction on validation dataset
    val1nn = knn_computions(X_train, y_train, X_val, 1)
    val3nn = knn_computions(X_train, y_train, X_val, 3)
    val5nn = knn_computions(X_train, y_train, X_val, 5)

    # logistic regression on validation dataset
    trained_weights = logistic_regression_training(
        X_train, y_train, alpha, max_iters, random_seed)
    valLogisticRegression = logistic_regression_prediction(
        X_val, trained_weights, threshold)

    # calculate the accuracy for each model
    val_accuracy = []
    val_accuracy.append(
        (y_val.flatten() == val1nn.flatten()).sum() / y_val.shape[0])
    val_accuracy.append(
        (y_val.flatten() == val3nn.flatten()).sum() / y_val.shape[0])
    val_accuracy.append(
        (y_val.flatten() == val5nn.flatten()).sum() / y_val.shape[0])
    val_accuracy.append(
        (y_val.flatten() == valLogisticRegression.flatten()).sum() / y_val.shape[0])

    # select the best method
    methods = ['1nn', '3nn', '5nn', 'logistic regression']
    best_method = methods[np.argmax(val_accuracy)]
    
    # calculate the accuracy of the test dataset using the best method
    X_train_val_merge = np.vstack([X_train, X_val])
    y_train_val_merge = np.vstack([y_train, y_val])
    test_accuracy = test_accuracy_computions(
        X_train_val_merge, y_train_val_merge, X_test, y_test, best_method, alpha, max_iters, random_seed, threshold)
    print(best_method, val_accuracy, test_accuracy)
    return best_method, val_accuracy, test_accuracy
  