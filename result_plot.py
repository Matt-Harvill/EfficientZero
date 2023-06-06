import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# CSV file path
csv_file = 'data.csv'

# Dictionary to store the extracted data
data = defaultdict(lambda: {'search_paths': set(), 'games': defaultdict(list)})

# Read the data from the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        title = row['Title']
        search_paths = int(row['Search Paths'])
        game = row['Game']
        mean_score = float(row['Mean Score'])
        data[title]['search_paths'].add(search_paths)
        data[title]['games'][game].append((search_paths, mean_score))

# Sort the data dictionary by title and search paths
sorted_data = dict(sorted(data.items()))
for title in sorted_data:
    sorted_data[title]['games'] = dict(sorted(sorted_data[title]['games'].items()))

# Plotting
for title, scores in sorted_data.items():
    search_paths = sorted(scores['search_paths'])
    games = scores['games']
    num_games = len(games)
    num_search_paths = len(search_paths)
    
    # Check if only one distinct number of search paths
    if len(search_paths) == 1:
        legend = False
    else:
        legend = True
    
    plt.figure(figsize=(10, 6))
    x = range(num_search_paths)
    width = 0.2
    
    # Sort the mean scores for each game by increasing search paths
    sorted_scores = []
    for game, mean_scores in games.items():
        sorted_scores.append(sorted(mean_scores, key=lambda x: x[0]))  # Sort by increasing search paths
    
    print(sorted_scores)

    # Plot Mean Scores for each game
    handles = []
    labels = []
    colors = ['#FFB6C1', '#FF69B4', '#FF1493', '#C71585']
    offset = (num_search_paths - 1) * width / 2
    for i, game in enumerate(games.keys()):
        mean_scores = [x[1] for x in sorted_scores[i]]
        search_paths = [x[0] for x in sorted_scores[i]]

        print(f"Game: {game}, Search Paths: {search_paths}, Mean Score: {mean_scores}")
        x_pos = [(j * width) + i - offset for j in x]
        handles.append(plt.bar(x_pos, mean_scores, width, color=colors))
        labels.append(game)
    
    plt.xlabel('Game')
    plt.ylabel('Mean Score')
    plt.title(f'Mean Scores for "{title}"')
    
    # Set x-axis labels and tick positions
    plt.xticks(range(num_games), games.keys())
    plt.xticks(rotation=45, ha='right')
    
    # Create legend with search paths information
    if legend:
        legend_colors = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        plt.legend(legend_colors, search_paths, title='Search Paths')
    
    # Show the plot
    plt.show()
