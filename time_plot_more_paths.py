import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
data = pd.read_csv('times_more_paths.csv')

# Extract relevant columns
groups = data['group']
mean_values = data['mean']
std_values = data['std']

# Calculate the number of data points
num_data_points = len(data)

colors = ['#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD', '#8E44AD', '#7D3C98', '#6C3483', '#5B2C6F', '#4A235A', '#1F618D'],


# Create the bar plot
fig, ax = plt.subplots()
axis_bars = ax.bar(range(num_data_points), mean_values, yerr=std_values, capsize=4)
axis_bars[0].set_color('#E8DAEF')
axis_bars[1].set_color('#D2B4DE')
axis_bars[2].set_color('#BB8FCE')
axis_bars[3].set_color('#A569BD')
axis_bars[0].set_edgecolor('black')
axis_bars[1].set_edgecolor('black')
axis_bars[2].set_edgecolor('black')
axis_bars[3].set_edgecolor('black')

# Set x-axis ticks and labels
ax.set_xticks(range(num_data_points))
ax.set_xticklabels(groups)

# Set labels and title
# ax.set_xlabel('Groups')
ax.set_ylabel('Mean Time (s)', fontweight='bold')
ax.set_title('Mean Time of 50 Simulations for More Paths', fontweight='bold')

# Create legend
# ax.legend(['Group'])
plt.xticks(rotation=45, ha='right', fontweight='bold')

# Adjust subplot parameters to add margin at the bottom
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(bottom=0.25)

# Display the plot
# plt.show()
# input("Proceed if you want to save the plot. Press Ctrl + C to exit.")
plt.savefig('images/More Paths Time Plot.png')