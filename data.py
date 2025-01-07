#Here is the goal to get the data from a plain text file
import os

dataset_folder = os.path.join(os.getcwd(), 'dataset')
all_files = os.listdir(dataset_folder)

print(all_files)
