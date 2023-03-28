import numpy as np
import pandas as pd
import sklearn.multioutput
from sklearn.svm import SVR


def calculate_accuracy_metric(preds: list[list[float, float]], labels: list[list[float, float]],
                              max_diff: float) -> float:
    """
        Returns prediction accuracy metric, based on given max difference

                Parameters:
                        preds (list[list[float, float]]): List of pairs of predicted values
                        labels (list[list[float, float]]): List of pairs of expected values
                        max_diff (float): Max difference value, for which we count prediction as correct

                Returns:
                        acc_metric (float): Accuracy metric value (good predictions / all predictions)
    """
    true_preds = 0
    num_preds = 0
    for i in range(len(preds)):
        pred_val = preds[i][0]  # valence
        pred_ar = preds[i][1]  # arousal

        label_val = labels[i][0]
        label_ar = labels[i][1]
        if (np.abs(pred_ar - label_ar) <= max_diff) and (np.abs(pred_val - label_val) <= max_diff):
            true_preds += 1
        num_preds += 1

    acc_metric = true_preds / num_preds
    return acc_metric


if __name__ == "__main__":
    df_path = "../data/DEAM/1_dnn_averaged/dataframe.pkl"
    columns_to_drop = ["song_id", " valence_std", " arousal_std"]
    test_fraction = 0.2
    random_state = 200

    data = pd.read_pickle(df_path)
    data = data.drop(columns=columns_to_drop)
    data[data.columns[[0, 1]]] = data[data.columns[[0, 1]]].div(10)  # scale labels to [0.1-0.9]

    test = data.sample(frac=test_fraction, random_state=random_state)
    train = data.drop(test.index)

    features = train[train.columns[2:]]  # normalize all, except labels
    train[train.columns[2:]] = (features - features.mean()) / features.std()

    test[test.columns[2:]] = (test[test.columns[2:]] - features.mean()) / features.std()

    train_data = train[train.columns[2:]]
    train_labels = train[train.columns[:2]]  # valence, arousal
    test_data = test[test.columns[2:]]
    test_labels = test[test.columns[:2]]

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    max_diff_values = [0.01, 0.03, 0.05]

    for kernel in kernels:
        regressor = sklearn.multioutput.MultiOutputRegressor(SVR(kernel=kernel, degree=3, gamma='scale', coef0=0.0, tol=1e-3, C=0.8, epsilon=0.05))
        regressor.fit(train_data, train_labels)

        preds_eval = regressor.predict(test_data)

        preds_train = regressor.predict(train_data)

        for max_diff in max_diff_values:
            acc_eval = calculate_accuracy_metric(preds_eval.tolist(), test_labels.values.tolist(), max_diff)
            print(f"Accuracy_{max_diff} of svm model with {kernel} kernel on 1_dnn dataframe evaluation set: ",
                  acc_eval)

            acc_train = calculate_accuracy_metric(preds_train.tolist(), train_labels.values.tolist(), max_diff)
            print(f"Accuracy_{max_diff} of svm model with {kernel} kernel on 1_dnn dataframe train set: ", acc_train)
