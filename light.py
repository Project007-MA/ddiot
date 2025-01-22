import lightgbm as lgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Load the dataset
data = load_digits()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare the LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters for LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss'
}

# Define a callback for early stopping and logging
early_stopping_callback = lgb.early_stopping(stopping_rounds=10)
log_callback = lgb.log_evaluation(period=1)

# Use record_evaluation to capture evaluation metrics
evaluation_results = {}

# Train the model with evaluation logging
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,  # analogous to "epochs"
    valid_sets=[train_data, val_data],  # Include both training and validation datasets
    valid_names=['train', 'validation'],  # Naming the datasets
    callbacks=[early_stopping_callback, log_callback, lgb.record_evaluation(evaluation_results)]
)

# Plot the log loss over boosting rounds (epochs)
plt.plot(evaluation_results['train']['multi_logloss'], label='Train Loss')
plt.plot(evaluation_results['validation']['multi_logloss'], label='Validation Loss')
plt.xlabel('Boosting Rounds (Epochs)')
plt.ylabel('Log Loss')
plt.title('Learning Curve for LightGBM')
plt.legend()
plt.show()
