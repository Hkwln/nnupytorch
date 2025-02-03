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
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda a: ''.join([int_to_char[i] for i in a])

#preparing the data for the model
data = torch.tensor(encode(all_text), dtype=torch.int64)

# 85% of the data is used for training, rest for testing
n = int(0.85 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is{context} the target:{target}")


batch_size = 4
block_size = 5

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        print(f"Batch {b}, Token {t}: input={xb[b, t]} target={yb[b, t]}")