from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
# Load the dataset
data = load_digits()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=45)

# Track accuracy with increasing number of estimators (trees)
estimators = list(range(10, 110, 10))
train_accuracies = []
val_accuracies = []

for n in estimators:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Plot the accuracy over increasing trees (analogous to epochs)
plt.plot(estimators, train_accuracies, label='Train Accuracy')
plt.plot(estimators, val_accuracies, label='Validation Accuracy')
plt.xlabel('Number of Trees (Estimators)')
plt.ylabel('Accuracy')
plt.title('Random Forest Learning Curve')
plt.legend()
plt.show()
