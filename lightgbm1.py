import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
from sklearn.preprocessing import MinMaxScaler
import numpy as np
data= pd.read_csv("Dataset/IoT1 Dataset/Model1.csv")
print(data.columns)
print(data[" Label"].value_counts())
data= pd.read_csv("Dataset/IoT1 Dataset/Model2.csv")
print(data.columns)
print(data[" Label"].value_counts())
data= pd.read_csv("Dataset/IoT1 Dataset/Model3.csv")
print(data.columns)
print(data[" Label"].value_counts())

print(data.shape)
data.dropna(axis=1,inplace=True)
data.drop_duplicates(inplace=True)
print(data.shape)
print(data.dtypes)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
data[" Label"]=le.fit_transform(data[" Label"])
print(data.shape)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
#y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# Optionally, you can also remove rows with NaN values, but it's better to impute them
data.dropna(inplace=True)
data.dropna(inplace=True)
#scaler=MinMaxScaler()
#data=scaler.fit_transform(data)
features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values

X_train, X_val, y_train, y_val = train_test_split(features,labels, test_size=0.2, random_state=42)

# Prepare the LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters for LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 3,
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
plt.figure(figsize=(10, 6))
# Plot the log loss over boosting rounds (epochs)
plt.plot(evaluation_results['train']['multi_logloss'], label='Train Loss')
plt.plot(evaluation_results['validation']['multi_logloss'], label='Validation Loss')
plt.xlabel('Iteration (Epochs)',)
plt.ylabel('Log Loss')
plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')
plt.legend()
plt.savefig("Results/lightgbm_graph",dpi=1000)
plt.show()

