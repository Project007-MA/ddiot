import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from save_load import *
from confu import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16
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
#y_train.dropna(inplace=True)


#scaler=MinMaxScaler()
#data=scaler.fit_transform(data)
features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values

# Load the MNIST-like dataset
#data = load_digits()
#X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# LightGBM client model
def train_lightgbm(X_train, y_train):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss'}
    model = lgb.train(params, train_data, num_boost_round=50)
    return model

# XGBoost client model
def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss'}
    model = xgb.train(params, dtrain, num_boost_round=50)
    return model

# RandomForest client model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Train individual client models
lightgbm_model = train_lightgbm(X_train, y_train)
xgboost_model = train_xgboost(X_train, y_train)
random_forest_model = train_random_forest(X_train, y_train)

# Create predictions on the validation set
lightgbm_predictions = np.argmax(lightgbm_model.predict(X_val), axis=1)
xgboost_predictions = np.argmax(xgboost_model.predict(xgb.DMatrix(X_val)), axis=1)
random_forest_predictions = random_forest_model.predict(X_val)



# Calculate accuracy for each local model
lightgbm_accuracy = accuracy_score(y_val, lightgbm_predictions)

unique_classes, class_counts = np.unique(lightgbm_predictions, return_counts=True)

# Print the result
for class_label, count in zip(unique_classes, class_counts):
    print(f"Class {class_label}: {count} instances")
res1=multi_confu_matrix(y_val,lightgbm_predictions)

save("res1",res1)
xgboost_accuracy = accuracy_score(y_val, xgboost_predictions)
res2=multi_confu_matrix(y_val,xgboost_predictions)

save("res2",res2)
random_forest_accuracy = accuracy_score(y_val, random_forest_predictions)
res3=multi_confu_matrix(y_val,random_forest_predictions)

save("res3",res3)


# Aggregating predictions (Majority Voting)
final_predictions = []
for i in range(len(y_val)):
    preds = [lightgbm_predictions[i], xgboost_predictions[i], random_forest_predictions[i]]
    final_predictions.append(np.bincount(preds).argmax())

# Accuracy of the ensemble model
ensemble_accuracy = accuracy_score(y_val, final_predictions)
res4=multi_confu_matrix(y_val,final_predictions)

save("res4",res4)

def res_plot1():
    mat = confusion_matrix(y_val, lightgbm_predictions)

    plt.figure(figsize=(10, 6))
    sns.heatmap(mat, annot=True, cmap="tab20b", fmt="d", xticklabels=["BENIGN", "Bot", "DDoS"],
                yticklabels=["BENIGN", "Bot", "DDoS"])

    plt.xlabel('Predicted labels',fontsize=23, fontweight='bold')
    plt.ylabel('True labels',fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/lightgbm_predictions", dpi=1000)
    plt.show()


res_plot1()

def res_plot2():
    mat = confusion_matrix(y_val, xgboost_predictions)

    plt.figure(figsize=(10, 6))
    sns.heatmap(mat, annot=True, cmap="summer", fmt="d", xticklabels=["BENIGN", "Bot", "DDoS"],
                yticklabels=["BENIGN", "Bot", "DDoS"])

    plt.xlabel('Predicted labels', fontsize=23, fontweight='bold')
    plt.ylabel('True labels', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/xgboost_predictions", dpi=1000)
    plt.show()


res_plot2()
def res_plot3():
    mat = confusion_matrix(y_val, random_forest_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(mat, annot=True, cmap="tab20c", fmt="d", xticklabels=["BENIGN", "Bot", "DDoS"],
                yticklabels=["BENIGN", "Bot", "DDoS"])

    plt.xlabel('Predicted labels', fontsize=23, fontweight='bold')
    plt.ylabel('True labels', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/random_forest_predictions.png", dpi=1000)
    plt.show()


res_plot3()

def res_plot4():
    mat = confusion_matrix(y_val,final_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(mat, annot=True, cmap="copper", fmt="d", xticklabels=["BENIGN","Bot","DDoS"],yticklabels=["BENIGN","Bot","DDoS"])

    plt.xlabel('Predicted labels', fontsize=23, fontweight='bold')
    plt.ylabel('True labels', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/final_predictions.png", dpi=1000)
    plt.show()

res_plot4()


