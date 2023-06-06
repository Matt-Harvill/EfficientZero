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
    group = scores['group']
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
    width = 0.6 / num_group
    
    # Sort the mean scores for each game by increasing search paths
    sorted_scores = []
    for game, mean_scores in games.items():
        # sorted_scores.append(sorted(mean_scores, key=lambda x: x[0]))  # Sort by increasing search paths
        sorted_scores.append(mean_scores)
    
    # Plot Mean Scores for each game
    handles = []
    labels = []

    color_arrays = [
        ['#D1F2EB', '#A3E4D7', '#76D7C4', '#48C9B0', '#1ABC9C', '#17A589', '#148F77', '#117864', '#0E6251', '#0B5345'],
        ['#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#2874A6', '#21618C', '#1B4F72', '#154360'],
        ['#FADBD8', '#F5B7B1', '#F1948A', '#EC7063', '#E74C3C', '#CB4335', '#B03A2E', '#943126', '#78281F', '#641E16'],
        ['#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD', '#8E44AD', '#7D3C98', '#6C3483', '#5B2C6F', '#4A235A', '#1F618D'],
        ['#FFCCCC', '#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000', '#990000', '#660000', '#330000', '#000000'],
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
        handles.append(plt.bar(x_pos, mean_scores, width, color=colors, edgecolor='black', linewidth=1))
        labels.append(game)
    
    plt.xlabel('Game', fontweight='bold')
    plt.ylabel('Mean Score', fontweight='bold')
    plt.title(f'Mean Scores for {title}', fontweight='bold')
    
    # Set x-axis labels and tick positions
    plt.xticks(range(num_games), games.keys())
    # plt.xticks(ha='right')
    # plt.xticks(rotation=45, ha='right')

    # Create legend with search paths information
    if legend:
        legend_colors = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        plt.legend(legend_colors, group, edgecolor='black')
        # plt.legend(legend_colors, group, title='Group')
    
    # Show the plot
    # plt.show()
    plt.savefig(f'images/{title}.png')