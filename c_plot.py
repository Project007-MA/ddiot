import matplotlib.pyplot as plt
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
# Data from the table
models = ['LSTM', 'CNN-Geo', 'ESN', 'Proposed Method']
accuracy = [96.44, 99.7,  98.98, 99.87]
precision = [95.74, 99.6,  98.27, 99.8]
recall = [97.66, 99.4, 98.62, 99.8]

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(models, accuracy, label='Accuracy', marker='<',linestyle=':')
plt.plot(models, precision, label='Precision', marker='>',linestyle='-.')
plt.plot(models, recall, label='Recall', marker='D',linestyle='--')
plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')

# Adding labels and title
plt.xlabel('Methods',fontsize=23,fontweight='bold')
plt.ylabel('Values',fontsize=23,fontweight='bold')
plt.legend()
# Display the plot
plt.tight_layout()
plt.grid()
plt.savefig("Results/c_plot1",dpi=1000)
plt.show()

# Data from the table
methodologies = ['CNN', 'ML-AIDS', 'Information Gain', 'RTIDS', 'FL']
accuracy_values = [99.7, 99.31, 96.48, 99.35, 99.87]

# Plotting the line graph
plt.figure(figsize=(8, 6))
plt.plot(methodologies, accuracy_values, label='Accuracy', marker='D', color='r',linestyle='-.')

plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')

plt.xlabel('Methods',fontsize=23,fontweight='bold')
plt.ylabel('Accuracy',fontsize=23,fontweight='bold')
# Display the plot
plt.tight_layout()
plt.grid()
plt.savefig("Results/c_plot2",dpi=1000)
plt.show()



# Data for plotting
methods = ['CNN', 'DNN', 'SVM', 'BiLSTM', 'FL']
accuracy = [99.50, 93.74, 97.20, 82.30, 99.87]

# Creating a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=methods, y=accuracy, palette="Blues_d")

plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')

plt.xlabel('Methods',fontsize=23,fontweight='bold')
plt.ylabel('Accuracy',fontsize=23,fontweight='bold')

plt.ylim(80, 100)  # Ensure y-axis is scaled from 0 to 100
plt.savefig("Results/c_plot3",dpi=1000)
plt.show()
