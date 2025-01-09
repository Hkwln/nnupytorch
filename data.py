#Here is the goal to get the data from a plain text file
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np


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
char_to_int ={char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
encode = np.array([char_to_int[char] for char in all_text], dtype=np.uint8)

print(encode("Guten Morgen"))