import os
import json, csv
from itertools import zip_longest

# Specify the folder path where the .mp4 files are located
folderPath = "UPDATE_ME/Videoannotation//Data//Video annotation"
# Actions to look for in the notes.txt file
actions = ['Inert atmosphere',
           'Addition of liquids',
           'Reflux',
           'Extraction',
           'Filtration',
           'Stir',
           'No Action'
           ]

# Function to count .mp4 files in the specified folder
def count_mp4_files(folder_path):
    '''
    Count the number of .mp4 files in the given folder.
    Args:
        folder_path (str): Path to the folder.
    Returns:
        int: Number of .mp4 files.
        list: Unique file names.
        list: File names with time stamps.
    '''
    # List to store .mp4 file names, the length of the list is the count of .mp4 files
    files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.mp4'):
            files.append(file[16:])
    # Get unique file names so having Cam1 and Cam2 for the same time is not double counted
    unique_list = []
    for item in files:
        if item not in unique_list:
            unique_list.append(item)
    files_list = []
    for f in os.listdir(folder_path):
        if f.lower().endswith('.mp4'):
            if f not in files_list:
                files_list.append(f[5:24])
    return len(files), unique_list, files_list



# mp4_count = count_mp4_files(folder)

# Gets the number of cleaned actions from the specified notes.txt 
def split_text_by_bracket(file_path):
    '''
    Split the text file content by '[' and filter parts based on specified actions.
    Args:
        file_path (str): Path to the text file.
    Returns:
        int: Number of filtered parts.
        list: Filtered parts containing specified actions.
    '''
    try:
        with open(file_path, 'r', encoding="latin-1") as f:
            content = f.read()
        # Split content by '['
        parts = content.split('[')
        filtered_parts = []
        for part in parts:
            # Check if any action is present in the part
            if any(']: ' + s in part for s in actions):
                # add '[' back to the part
                filtered_parts.append("[" + part)  
        return len(filtered_parts), filtered_parts
    except FileNotFoundError:
        print("File not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Main execution

# Get folder name from user input
folder = os.path.join(folderPath , input("Enter the name of the data folder:"))

# Dictionary to store data analysis results
data_analysis = {}
# Iterate through items in the specified folder
for item in os.listdir(folder):
    full_path = os.path.join(folder, item)
    # check if it's a folder and starts with '202' (2025, to avoid eventually other, obsolete folders)
    if os.path.isdir(full_path) and item[0:3] == '202':  
        print("Folder:", item)
        if os.path.isdir(full_path):
            # Count .mp4 files in the folder
            mp4_count = count_mp4_files(full_path)[0]
            mp4_times = count_mp4_files(full_path)[1]
            print(f"Number of .mp4 files in '{full_path}': {mp4_count}")
            try:
                totlabels = split_text_by_bracket(os.path.join(full_path, 'notes.txt'))[0]
                labels = split_text_by_bracket(os.path.join(full_path, 'notes.txt'))[1]
            except IndexError:
                pass
            # Store the analysis results in the dictionary
            data_analysis[item] = [f'TotLabels = {totlabels}', 
                                   f'Videos = {mp4_count}', 
                                   2*totlabels == mp4_count, 
                                   labels, 
                                   mp4_times]
            if 2 * totlabels != mp4_count:
                with open(os.path.join(full_path, "table.csv"), "w", newline="", encoding="latin-1") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Videos", "Labels"])  # header row
                    for a, b in zip_longest(mp4_times, labels, fillvalue=""):
                        writer.writerow([a, b])
            ########## COMMENT TO CHECK THE JSON ONLY, UNCOMMENT TO CHANGE TIMES (MAKE JSONL) AFTER MANUAL CORRECTIONS
            # elif 2*totlabels == mp4_count:
            #     newlabels = labels.copy()
            #     for i, e in enumerate(mp4_times):
            #         e = e.replace("-", ":")
            #         newlabels[i] = labels[i][:12] + e[0:8] + labels[i][20:]
            #     with open(os.path.join(full_path, "notes.txt"), "w") as f:
            #         for line in newlabels:
            #             f.write(line + "\n")
            #     print("Updated notes.txt with corrected timestamps.")
            #     with open(os.path.join(full_path,"labels.jsonl"), "w", encoding="utf-8") as f:
            #         for n, item in enumerate(labels):
            #             print(count_mp4_files(full_path)[2][n], item)
            #             json.dump({"image":count_mp4_files(full_path)[2][n], "label":item.split("]:", 1)[1].split(":", 1)[0].strip()}, f)
            #             f.write("\n")
            #     print("Created labels.jsonl file.")

        else:
            print("Invalid folder path.")

# Save the data analysis results to a JSON file
with open("data.json", "w") as f:
    json.dump(data_analysis, f, indent=4)