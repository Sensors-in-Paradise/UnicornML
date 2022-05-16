import os
import numpy as np
from evaluation.metrics import accuracy_threshold

def create_text_metrics(path: str, y_test_pred: np.ndarray, y_test_true: np.ndarray, metric_funcs: list) -> None:
    """
    TODO: not working a the moment with more than one function, because the data gets changed (passed by reference)
    - make metric functions copy the data

    executes metric funcs  and writes the results in a text file with the provided path
    """
    with open(os.path.join(path, 'metrics.txt'), "w") as f:
        for metric_func in metric_funcs:
            f.write(f"{metric_func.__name__}: {metric_func(y_test_pred, y_test_true)}\n")
            
def create_text_metrics_for_threshold_accuracy(path: str, y_test_pred: np.ndarray, y_test_true: np.ndarray, thresholds: 'list[Float]') -> None:

    with open(os.path.join(path, 'metrics.txt'), "a") as f:
        for threshold in thresholds:
            acc, percentage_lost = accuracy_threshold(y_test_pred, y_test_true, threshold)
            f.write(f"accuracy with threshold={threshold}: {round(acc, 4)}, but was unable to predict {round(percentage_lost, 2)*100} %\n")