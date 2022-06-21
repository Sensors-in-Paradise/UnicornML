"""
functions that take 
- the prediction_vectors ([0.03, 0.5, 0.3], [0.03, 0.5, 0.3], ...) 
- and y_test ([0, 1, 0, 0], [0, 0, 0, 1], ...)

and give you metrics like accuracy, f1_score and MUCH MORE ... <3

PS: 
if we would execute np.argmax before, infos like the model was not sure would be lost
"""

import numpy as np
from sklearn.metrics import f1_score

def f1_score(prediction_vectors: np.ndarray, y_test: np.ndarray) -> float:
    prediction_vectors = np.argmax(prediction_vectors, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return f1_score(y_test, prediction_vectors)

def accuracy(prediction_vectors: np.ndarray, y_test: np.ndarray, verbose: int = 0) -> float:
    predictions = np.argmax(prediction_vectors, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.sum(predictions == y_test) / len(predictions)
    if verbose:
        print(f"accuracy: {accuracy}")
    return accuracy

def accuracy_with_recording_context(prediction_vectors: np.ndarray, y_test: np.ndarray, verbose: int = 0) -> float:
    prediction_length = 0 
    correct_preds = 0
    for idx, prediction_array in enumerate(prediction_vectors):
        predictions = np.argmax(prediction_array, axis=1)
        y_test_pred = np.argmax(y_test[idx], axis=1)
        prediction_length += len(prediction_array)
        correct_preds += np.sum(predictions == y_test_pred) 
    return (correct_preds / prediction_length)
        

def accuracy_threshold(prediction_vectors: np.ndarray, y_test: np.ndarray, threshold: float, verbose: int = 0) -> float:

    total_predictions = 0
    correct_predictions = 0
    for idx, prediction in enumerate(prediction_vectors):
        prediction_idx = np.argmax(prediction)
        confidence = prediction[prediction_idx]
        if confidence < threshold:
            continue
        true_idx = np.argmax(y_test[idx])
        if(true_idx == prediction_idx):
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    if verbose:
        print(f"accuracy of confident predictions: {accuracy}")
        print("ignored "+len(y_test)+" predictions, which is equivalent to "+str(1-(total_predictions/len(y_test)))+"%")
        
    return accuracy, 1-(total_predictions/len(y_test))

def average_failure_rate(prediction_vectors: np.ndarray, y_test: np.ndarray) -> float:
    """
    output y_test [0.03, 0.5, 0.3], correct label idx 2
    -> how much is missing to 1.0?
    """

    label_indices = []
    failure_sum = 0

    # get indices of correct labels
    for i in range(len(y_test)):
        label_indices.append(np.argmax(y_test[i]))  # [2, 1, 0, 3, ...]

    # sum up failure rate by calculating "1 - the prediction value of row i and expected column"
    for i in range(len(label_indices)):
        failure_sum += 1 - prediction_vectors[i][label_indices[i]]

    average_failure_rate = failure_sum / len(label_indices)
    return average_failure_rate