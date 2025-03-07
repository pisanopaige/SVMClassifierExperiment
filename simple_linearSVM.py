from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load digits data
digits_data = load_digits()
x = digits_data.data
y = digits_data.target

# Filter the digits data so only two digits are identifiable
x, y = x[(y == 1) | (y == 9)], y[(y == 1) | (y == 9)]

# Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Train linear SVM classifier using train datasets
classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter = 1000, random_state = 42)
classifier.fit(x_train, y_train)

# Make predictions using test datasets and evaluate how well the model does
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

# Get weights and bias
weights = classifier.coef_
bias = classifier.intercept_

print(accuracy)
print(confusionMatrix)
print(weights)
print(bias)