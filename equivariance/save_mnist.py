import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
from scipy import ndimage

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.10, random_state=42)
X_train = tf.expand_dims(X_train, axis=-1)
X_val = tf.expand_dims(X_val, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)
X_test = np.array(X_test)
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
Y_test = Y_test.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
Y_val = Y_val.reshape(-1,1)
encoder.fit(Y_train)
Y_test = encoder.transform(Y_test).toarray()
Y_train = encoder.transform(Y_train).toarray()
Y_val = encoder.transform(Y_val).toarray()

#Rotate the test data up to 50 degrees
min_deg, max_deg = 0, 50
angles = np.random.randint(min_deg, max_deg, size=len(X_test))
for i, a in enumerate(angles):
  X_test[i,...,0] = ndimage.rotate(X_test[i,...,0], a, reshape=False)


np.save("data/X_test.npy", X_test)
np.save(f"data/Y_test.npy", Y_test)
np.save(f"data/X_val.npy", X_val)
np.save(f"data/Y_val.npy", Y_val)
np.save(f"data/X_train.npy", X_train)
np.save(f"data/Y_train.npy", Y_train)       