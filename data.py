#Here is the goal to get the data from a plain text file
import torch
import torch.nn as nn
from torch.nn import functional as F
import os


dataset_folder = os.path.join(os.getcwd(), 'dataset')
all_files = os.listdir(dataset_folder)

#filters out all non text files from the dataset foldre
text_files = [file for file in all_files if file.endswith('.txt')]

# Read the contents of each text file and concatenate them
all_text = ""
for file_name in text_files:
    with open(os.path.join(dataset_folder, file_name), 'r', encoding='UTF-8') as file:
        all_text += file.read()
        print(len(all_text))

#here are all the unique charracters that occur in this text:
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(vocab_size)
print(chars)
# Create a mapping from characters to integers
