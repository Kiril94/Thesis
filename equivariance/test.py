import numpy as np
from sklearn.metrics import f1_score

y_pred = np.array([0.1, 0.6, 0.8])
y_true = np.array([1, 1, 0])
y_true = np.expand_dims(y_true, axis =1 )
y_pred = np.expand_dims(y_pred, axis =1 )
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0
f1 = f1_score(y_pred, y_true, average='micro')
print(np.concatenate((y_true, y_pred), axis=-1).astype(int).shape)

