import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data for the bar chart
labels = ['Poisoned Update (No Encryption)', 'Poisoned Update (With Encryption)']
values = [40, 0]  # Percentages as shown in the image

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, width=0.2, color='blue')  # Set bar width to 0.2

# Add labels and title
plt.ylabel('Percentage (%)')
plt.title('Attack Prevention with Homomorphic Encryption')
plt.ylim(0, 50)  # Setting y-axis limit to make 0% stand out more

# Display values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval}%", ha='center', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
