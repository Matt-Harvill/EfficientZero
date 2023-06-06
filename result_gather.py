import os
import re
import csv

# Directory path containing the test.log files
directory = 'Final_Results'

# CSV file path to store the extracted data
csv_file = 'test_scores.csv'

# Regular expression patterns for extracting the desired information
mean_pattern = r'Test Mean Score: (\d+\.\d+)'
max_pattern = r'max: (\d+\.\d+)'
min_pattern = r'min: (\d+\.\d+)'
std_pattern = r'Test Std Score: (\d+\.\d+)'

# List to store the extracted data
data = []

# Recursive function to search for log files
def search_logs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename == 'test.log':
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    content = file.read()
                    mean_match = re.search(mean_pattern, content)
                    max_match = re.search(max_pattern, content)
                    min_match = re.search(min_pattern, content)
                    std_match = re.search(std_pattern, content)
                    if mean_match and max_match and min_match and std_match:
                        mean_score = mean_match.group(1)
                        max_score = max_match.group(1)
                        min_score = min_match.group(1)
                        std_score = std_match.group(1)
                        data.append([file_path, mean_score, max_score, min_score, std_score])

# Call the recursive function to search for log files in the directory tree
search_logs(directory)

# Write the extracted data to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Path', 'Mean Score', 'Max Score', 'Min Score', 'Std Score'])
    writer.writerows(data)
