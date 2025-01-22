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
data.dropna(axis=1,inplace=True)
data.drop_duplicates(inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
data[" Label"]=le.fit_transform(data[" Label"])

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
#y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# Optionally, you can also remove rows with NaN values, but it's better to impute them
data.dropna(inplace=True)
data.dropna(inplace=True)
#scaler=MinMaxScaler()
#data=scaler.fit_transform(data)
features=data.iloc[:200000,:-1].values
labels=data.iloc[:200000,-1].values



X_train, X_val, y_train, y_val = train_test_split(features,labels, test_size=0.2, random_state=45)

"""X_train = pd.DataFrame(X_train)  # Convert to DataFrame if necessary
X_val = pd.DataFrame(X_val)
#y_train=pd.DataFrame(y_train)
# Check for 'inf' values and replace them with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
#y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# Optionally, you can also remove rows with NaN values, but it's better to impute them
X_train.dropna(inplace=True)
X_val.dropna(inplace=True)
#y_train.dropna(inplace=True)
# Track accuracy with increasing number of estimators (trees)"""
estimators = list(range(10, 110, 10))
from sklearn.metrics import log_loss

# Initialize lists to track loss
train_losses = []
val_losses = []

for n in estimators:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)

    # Get predictions (predict_proba is used for calculating log loss)
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)

    # Calculate log loss
    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plotting the loss over increasing trees (analogous to epochs)
plt.figure(figsize=(10, 6))
plt.plot(estimators, train_losses, label='Train Loss', color='blue', linestyle='--')
plt.plot(estimators, val_losses, label='Validation Loss', color='red', linestyle='-')

plt.xlabel('Number of Trees (Estimators)', fontsize=23, fontweight='bold')
plt.ylabel('Log Loss', fontsize=23, fontweight='bold')
plt.xticks(fontsize=23, fontweight='bold')
plt.yticks(fontsize=23, fontweight='bold')
plt.legend()

# Save the figure
plt.savefig("Results/random_forest_loss_graph", dpi=1000)

# Show the plot
plt.show()

