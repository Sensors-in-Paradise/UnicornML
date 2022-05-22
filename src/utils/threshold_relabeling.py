import numpy as np

def relabel_by_threshold_with_recording_context(original_prediction_list, threshold, relabel_func):
    relabeled_prediction_list = []
    for original_predictions in original_prediction_list:
        relabeled_prediction_list += relabel_by_threshold(original_predictions, threshold, relabel_func)
    return [item for sublist in relabeled_prediction_list for item in sublist] # flattened by one dimension, since we don't need the recording context anymore after that step
    

def relabel_by_threshold(original_predictions, threshold, relabel_func):
    def erase_labels_under_threshold(y_test_pred):
        if max(y_test_pred) >= threshold:
            return y_test_pred
        return []
    
    original_predictions_copy = original_predictions.copy()
    filtered_predictions = list(map(erase_labels_under_threshold, original_predictions_copy))
    assert len(original_predictions_copy) == len(filtered_predictions), "length of y_test_preds and y_test_filtered are not equal"
    relabeled = relabel_func(filtered_predictions, original_predictions)
    return relabeled
    
def relabel_feed_forward(filtered_predictions, original_predictions):
    relabeled_y_test_pred =  []
    for _ in range(len(filtered_predictions)):
        relabeled_y_test_pred.append([])
        
    for idx, pred in enumerate(filtered_predictions):
        if len(pred) == 0:
            
            # if prediction at index 0 is empty, take the original prediction 
            if idx == 0:
                relabeled_y_test_pred[idx] = original_predictions[idx]
            else:
                relabeled_y_test_pred[idx] = relabeled_y_test_pred[idx-1]
        else:
            relabeled_y_test_pred[idx] = filtered_predictions[idx]
    return relabeled_y_test_pred

def relabel_feed_backwards(filtered_predictions, original_predictions):
    
    # reverse input lists, do the same as feed forward and reverse again
    relabeled_y_test_pred =  []
    for _ in range(len(filtered_predictions)):
        relabeled_y_test_pred.append([])
        
    filtered_predictions = filtered_predictions[::-1]
    original_predictions = original_predictions[::-1]
    for idx, pred in enumerate(filtered_predictions):
        if len(pred) == 0:
            if idx == 0:
                relabeled_y_test_pred[idx] = original_predictions[idx]
            else:
                relabeled_y_test_pred[idx] = relabeled_y_test_pred[idx-1]
        else:
            relabeled_y_test_pred[idx] = filtered_predictions[idx]
            
    return relabeled_y_test_pred[::-1]

def relabel_by_confidence_interpolation(filtered_predictions, original_predictions):
    relabeled_y_test_pred = []

    for _ in range(len(filtered_predictions)):
        relabeled_y_test_pred.append([])
        
    # get indices of non empty lists in y_test_filtered
    non_empty_indices = [idx for idx, pred in enumerate(filtered_predictions) if len(pred) > 0]
    
    # if prediction at index 0 is empty, take the original prediction and insert that into non_empty_indices for interpolation
    if(len(filtered_predictions[0]) == 0):
        filtered_predictions[0] = original_predictions[0]
        non_empty_indices.insert(0,0)
    
    index_counter = 1
    previous_non_empty_pred = filtered_predictions[0]
    next_non_empty_pred = filtered_predictions[non_empty_indices[index_counter]]
    
    for idx, pred in enumerate(filtered_predictions):
        
        # if empty, take the maximum confidence value of the previous non empty argmax and next non empty argmax
        if len(pred) == 0:
            relabeled_y_test_pred[idx] = previous_non_empty_pred if original_predictions[idx][np.argmax(previous_non_empty_pred)] > original_predictions[idx][np.argmax(next_non_empty_pred)] else next_non_empty_pred
        
        # if not empty, copy the predictions and reset previous_non_empty_pred and next_non_empty_pred
        else:
            index_counter += 1
            previous_non_empty_pred = pred
            if index_counter < len(non_empty_indices):
                next_non_empty_pred = filtered_predictions[non_empty_indices[index_counter]]
            relabeled_y_test_pred[idx] = pred
            
    return relabeled_y_test_pred
    
