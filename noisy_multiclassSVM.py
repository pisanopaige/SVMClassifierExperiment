from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load digits data
digits_data = load_digits()
x = digits_data.data
y = digits_data.target

# Define noise variances
noise_variances = [50, 100, 200, 400, 800]

# Define random generator for noise
random = np.random.default_rng(42)

# Add noise and evaluate how well the model does for each variance
for var in noise_variances:
    # Define random Gaussian noise
    noise = random.normal(loc=0, scale=np.sqrt(var), size=x.shape)

    # Add noise
    x_noise = x + noise

    # Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x_noise, y, test_size=0.2, random_state=42)

    # Train SVM classifier using train datasets
    classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42)
    classifier.fit(x_train, y_train)

    # Make predictions using test datasets and evaluate how well the model does
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)

    print(var)
    print(accuracy)
    print(confusionMatrix)