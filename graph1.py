import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data
techniques = ['No Compression', 'Quantization', 'Sparsification', 'Quantization + Sparsification']
update_sizes = [5.0, 2.0, 1.5, 0.8]

# Plotting
plt.figure(figsize=(8, 5))

# Plot each technique as a separate line
for i in range(len(techniques) - 1):
    plt.plot(
        [techniques[i], techniques[i + 1]],
        [update_sizes[i], update_sizes[i + 1]],
        marker='o',
        linestyle='-',
        color='blue',
        linewidth=2
    )

plt.ylabel('Update Size (MB)')
plt.title('Communication Overhead Reduction')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
