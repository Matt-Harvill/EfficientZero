import matplotlib.pyplot as plt
import csv

# CSV file path
csv_file = 'test_scores.csv'

# Lists to store the extracted data
titles = []
games = []
mean_scores = []
max_scores = []
min_scores = []
std_scores = []

# Read the data from the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    i = 0
    for row in reader:
        if i == 9:
            break
        titles.append(row[0])
        games.append(row[1])
        mean_scores.append(float(row[2]))
        max_scores.append(float(row[3]))
        min_scores.append(float(row[4]))
        std_scores.append(float(row[5]))
        i += 1

# Plotting
plt.figure(figsize=(10, 6))
x = range(len(titles))
width = 0.2

# Plot Mean Scores
plt.bar(x, mean_scores, width, label='Mean Score')
# Plot Max Scores
plt.bar([i + width for i in x], max_scores, width, label='Max Score')
# Plot Min Scores
plt.bar([i + 2 * width for i in x], min_scores, width, label='Min Score')
# Plot Std Scores
plt.bar([i + 3 * width for i in x], std_scores, width, label='Std Score')

# Set x-axis labels and tick positions
plt.xticks([i + 1.5 * width for i in x], titles)
plt.xlabel('Title')
plt.ylabel('Scores')

# Set the title of the plot
plt.title('Scores by Title')

# Display the legend
plt.legend()

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the layout to prevent overlapping of labels
plt.tight_layout()

# Show the plot
plt.show()
