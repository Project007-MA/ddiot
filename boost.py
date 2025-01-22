import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
# Load the dataset
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(type(X_test))
print(X_test.shape)
print(y_test.shape)
print(type(y_test))
print(y_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)

# Parameters for XGBoost
params = {
    'objective': 'multi:softprob',
    'num_class': 10,
    'eval_metric': 'mlogloss'
}

# Train the model with evaluation metric logging
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,  # analogous to "epochs"
    evals=[(dtrain, 'train'), (dval, 'validation')],
    evals_result=evals_result,
    early_stopping_rounds=10,  # stop early if no improvement
    verbose_eval=False
)

# Plot the log loss over boosting rounds (epochs)
plt.plot(evals_result['train']['mlogloss'], label='Train Loss')
plt.plot(evals_result['validation']['mlogloss'], label='Validation Loss')
plt.xlabel('Boosting Rounds (Epochs)')
plt.ylabel('Log Loss')
plt.title('Learning Curve for XGBoost')
plt.legend()
plt.show()




