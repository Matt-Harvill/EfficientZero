import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import random

# CSV file path
csv_file = 'data_cleaned.csv'

# Dictionary to store the extracted data
data = defaultdict(lambda: {'group': set(), 'games': defaultdict(list)})

# Read the data from the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        title = row['Title']
        group = row['Group']
        game = row['Game']
        mean_score = float(row['Mean Score'])
        data[title]['group'].add(group)
        data[title]['games'][game].append((group, mean_score))

# Sort the data dictionary by title and search paths
# sorted_data = dict(sorted(data.items()))
sorted_data = dict(data.items())
for title in sorted_data:
    # sorted_data[title]['games'] = dict(sorted(sorted_data[title]['games'].items()))
    sorted_data[title]['games'] = dict(sorted_data[title]['games'].items())

# Plotting
for title, scores in sorted_data.items():
    group = sorted(scores['group'])
    games = scores['games']
    num_games = len(games)
    num_group = len(group)
    
    # Check if only one distinct number of search paths
    if len(group) == 1:
        legend = False
    else:
        legend = True
    
    plt.figure(figsize=(10, 6))
    x = range(num_group)
    width = 0.2
    
    # Sort the mean scores for each game by increasing search paths
    sorted_scores = []
    for game, mean_scores in games.items():
        sorted_scores.append(sorted(mean_scores, key=lambda x: x[0]))  # Sort by increasing search paths
    
    # Plot Mean Scores for each game
    handles = []
    labels = []

    color_arrays = [
        ['#BDFCC9', '#87EBA2', '#55D982', '#1EA457'],
        ['#CCEBFF', '#80C1FF', '#4DA8FF', '#267EFF'],
        ['#FFD6E7', '#FFA3C2', '#FF80A6', '#FF5480'],
        ['#E8D7FF', '#BEA7FF', '#956DFF', '#6732FF'],
        ['#FFCCCC', '#FF8080', '#FF4D4D', '#FF2626'],
    ]

    # Generate a random index
    random_index = random.randint(0, len(color_arrays) - 1)
    # Select a random color array
    colors = color_arrays[random_index]

    offset = (num_group - 1) * width / 2
    for i, game in enumerate(games.keys()):
        mean_scores = [x[1] for x in sorted_scores[i]]
        group = [x[0] for x in sorted_scores[i]]

        # print(f"Game: {game}, Group: {group}, Mean Score: {mean_scores}")
        x_pos = [(j * width) + i - offset for j in x]
        handles.append(plt.bar(x_pos, mean_scores, width, color=colors))
        labels.append(game)
    
    plt.xlabel('Game')
    plt.ylabel('Mean Score')
    plt.title(f'Mean Scores for {title}')
    
    # Set x-axis labels and tick positions
    plt.xticks(range(num_games), games.keys())
    # plt.xticks(ha='right')
    # plt.xticks(rotation=45, ha='right')

    # Create legend with search paths information
    if legend:
        legend_colors = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        plt.legend(legend_colors, group)
        # plt.legend(legend_colors, group, title='Group')
    
    # Show the plot
    # plt.show()
    plt.savefig(f'images/{title}.png')
