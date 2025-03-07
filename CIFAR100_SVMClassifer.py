from tensorflow import keras
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Flatten the data
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()

# Train SVM classifier using train datasets
classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter = 1000, random_state = 42, n_jobs=-1)
classifier.fit(x_train_flat, y_train_flat)

# Make predictions using test datasets and evaluate how well the model does
y_pred = classifier.predict(x_test_flat)
accuracy = accuracy_score(y_test_flat, y_pred)
confusionMatrix = confusion_matrix(y_test_flat, y_pred)

print(accuracy)
print(confusionMatrix)