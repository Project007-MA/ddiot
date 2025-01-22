from save_load import *
from confu import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from save_load import *
from confu import *
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
def res1():
    # Define the metrics and their corresponding values
    metrics = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f_measure", "npv", "fpr", "fnr"
    ]
    metrics = [metric.capitalize() for metric in metrics]
    values =load("res1")
    values =[i*100 for i in values]
    # Create a DataFrame
    data = pd.DataFrame({'Metric': metrics[:-2], 'Value': values[:-2]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='tab20b')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.xlim(80,100)
    plt.tight_layout()
    plt.savefig("Results/lightgbm_predictions_res1",dpi=1000)
    plt.show()

    data = pd.DataFrame({'Metric': metrics[-2:], 'Value': values[-2:]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='tab20b')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/lightgbm_predictions_res2",dpi=1000)
    plt.show()


    data = {'Metric': metrics, 'Value': values}
    df = pd.DataFrame(data)
    df.to_csv("Results/lightgbm_predictions_res.csv")
    # Display the DataFrame
    print(df)
res1()

def res2():
    # Define the metrics and their corresponding values
    metrics = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f_measure", "npv", "fpr", "fnr"
    ]
    metrics = [metric.capitalize() for metric in metrics]
    values =load("res2")
    values =[i*100 for i in values]
    # Create a DataFrame
    data = pd.DataFrame({'Metric': metrics[:-2], 'Value': values[:-2]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='summer')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.xlim(80,100)
    plt.tight_layout()
    plt.savefig("Results/xgboost_predictions_res1",dpi=1000)
    plt.show()

    data = pd.DataFrame({'Metric': metrics[-2:], 'Value': values[-2:]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='summer')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/xgboost_predictions_res2",dpi=1000)
    plt.show()


    data = {'Metric': metrics, 'Value': values}
    df = pd.DataFrame(data)
    df.to_csv("Results/xgboost_predictions_res.csv")
    # Display the DataFrame
    print(df)
res2()

def res3():
    # Define the metrics and their corresponding values
    metrics = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f_measure", "npv", "fpr", "fnr"
    ]
    metrics = [metric.capitalize() for metric in metrics]
    values =load("res3")
    values =[i*100 for i in values]
    # Create a DataFrame
    data = pd.DataFrame({'Metric': metrics[:-2], 'Value': values[:-2]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='tab20c')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.xlim(80,100)
    plt.tight_layout()
    plt.savefig("Results/random_forest_predictions_res1",dpi=1000)
    plt.show()

    data = pd.DataFrame({'Metric': metrics[-2:], 'Value': values[-2:]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='tab20c')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/random_forest_predictions_res2",dpi=1000)
    plt.show()


    data = {'Metric': metrics, 'Value': values}
    df = pd.DataFrame(data)
    df.to_csv("Results/random_forest_predictions_res.csv")
    # Display the DataFrame
    print(df)
res3()

def res4():
    # Define the metrics and their corresponding values
    metrics = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f_measure", "npv", "fpr", "fnr"
    ]
    metrics = [metric.capitalize() for metric in metrics]
    values =load("res4")
    values =[i*100 for i in values]
    # Create a DataFrame
    data = pd.DataFrame({'Metric': metrics[:-2], 'Value': values[:-2]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='copper')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.xlim(80,100)
    plt.tight_layout()
    plt.savefig("Results/final_predictions_res1",dpi=1000)
    plt.show()

    data = pd.DataFrame({'Metric': metrics[-2:], 'Value': values[-2:]})

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Metric', data=data, palette='copper')

    plt.ylabel('Metrics', fontsize=23, fontweight='bold')
    plt.xlabel('Values', fontsize=23, fontweight='bold')
    plt.xticks(fontsize=23, fontweight='bold')
    plt.yticks(fontsize=23, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Results/final_predictions_res2",dpi=1000)
    plt.show()


    data = {'Metric': metrics, 'Value': values}
    df = pd.DataFrame(data)
    df.to_csv("Results/final_predictions_res.csv")
    # Display the DataFrame
    print(df)
res4()