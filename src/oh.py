# from sklearn.model_selection import StratifiedKFold, KFold

# X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# kf = KFold(n_splits=3, random_state=None)

# for train_index, test_index in kf.split(X):
#     print(train_index)
#     print(test_index)
#     print("______")

#print(kf.split(X))

import numpy as np

y_test_pred = np.array([[1,2], [2,3], [3,4], [4,5]])
confusion_y_test_pred = y_test_pred
confusion_y_test_pred = np.append(confusion_y_test_pred, y_test_pred, axis=0)
confusion_y_test_pred = np.append(confusion_y_test_pred, y_test_pred, axis=0)
confusion_y_test_pred = np.append(confusion_y_test_pred, y_test_pred, axis=0)

#print(type(y_test_pred))
print(len(y_test_pred))

# calc_metric = lambda v1, v2, v3: (860024*v1 + 952351*v2 + 858291*v3) / (860024+952351+858291)
# s = calc_metric(0.39279, 0.6092, 0.72512)
# s_p = calc_metric(0.40697045134754845, 0.6560398170567662, 0.7153970826580227)
# p = calc_metric(0.31442796839484793, 0.30669895076674736, 0.34683954619124796)

# print(s, s_p, p)