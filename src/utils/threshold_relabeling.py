import numpy as np

def relabel_by_threshold(y_test_pred_original, threshold, relabel_func):
    def erase_labels_under_threshold(y_test_pred):
        if max(y_test_pred) >= threshold:
            return y_test_pred
        return []
    
    y_test_preds = y_test_pred_original.copy()
    y_test_filtered = list(map(erase_labels_under_threshold, y_test_preds))
    assert len(y_test_preds) == len(y_test_filtered), "length of y_test_preds and y_test_filtered are not equal"
    relabeled = relabel_func(y_test_filtered, y_test_pred_original)
    return relabeled
    
def relabel_feed_forward(y_test_filtered, y_test_pred_original):
    relabeled_y_test_pred =  []
    for _ in range(len(y_test_filtered)):
        relabeled_y_test_pred.append([])
        
    for idx, pred in enumerate(y_test_filtered):
        if len(pred) == 0:
            if idx == 0:
                relabeled_y_test_pred[idx] = y_test_pred_original[idx]
            else:
                relabeled_y_test_pred[idx] = relabeled_y_test_pred[idx-1]
        else:
            relabeled_y_test_pred[idx] = y_test_filtered[idx]
    return relabeled_y_test_pred

def relabel_feed_backwards(y_test_filtered, y_test_pred_original):
    
    # reverse input lists, do the same as feed forward and reverse again
    relabeled_y_test_pred =  []
    for _ in range(len(y_test_filtered)):
        relabeled_y_test_pred.append([])
        
    y_test_filtered = y_test_filtered[::-1]
    y_test_pred_original = y_test_pred_original[::-1]
    for idx, pred in enumerate(y_test_filtered):
        if len(pred) == 0:
            if idx == 0:
                relabeled_y_test_pred[idx] = y_test_pred_original[idx]
            else:
                relabeled_y_test_pred[idx] = relabeled_y_test_pred[idx-1]
        else:
            relabeled_y_test_pred[idx] = y_test_filtered[idx]
            
    return relabeled_y_test_pred[::-1]

def relabel_by_confidence_interpolation(y_test_filtered, y_test_pred_original):
    relabeled_y_test_pred = []

    for _ in range(len(y_test_filtered)):
        relabeled_y_test_pred.append([])
        
    # get indices of non empty lists in y_test_filtered
    non_empty_indices = [idx for idx, pred in enumerate(y_test_filtered) if len(pred) > 0]
    
    # if prediction at index 0 is empty, take the original prediction and insert that into non_empty_indices for interpolation
    if(len(y_test_filtered[0]) == 0):
        y_test_filtered[0] = y_test_pred_original[0]
        non_empty_indices.insert(0,0)
    
    counter = 1
    previous_non_empty = y_test_filtered[0]
    next_non_empty = y_test_filtered[non_empty_indices[counter]]
    
    for idx, pred in enumerate(y_test_filtered):
        if len(pred) == 0:
            relabeled_y_test_pred[idx] = previous_non_empty if y_test_pred_original[idx][np.argmax(previous_non_empty)] > y_test_pred_original[idx][np.argmax(next_non_empty)] else next_non_empty
            
        else:
            counter += 1
            previous_non_empty = pred
            if counter < len(non_empty_indices):
                next_non_empty = y_test_filtered[non_empty_indices[counter]]
            relabeled_y_test_pred[idx] = pred
            
    return relabeled_y_test_pred
    
