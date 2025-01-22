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
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.3)

# Check for 'inf' values and replace them with NaN

# Prepare the DMatrix
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_test, label=y_test)

# Parameters for XGBoost
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
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
plt.figure(figsize=(10, 6))
# Plot the log loss over boosting rounds (epochs)
plt.plot(evals_result['train']['mlogloss'], label='Train Loss')
plt.plot(evals_result['validation']['mlogloss'], label='Validation Loss')
plt.xlabel('Iteration (Epochs)',fontsize=23,fontweight='bold')
plt.ylabel('Log Loss',fontsize=23,fontweight='bold')
plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')
plt.legend()
plt.savefig("Results/xg_boost_graph",dpi=1000)
plt.show()


