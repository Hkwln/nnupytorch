import torch
from datasets import load_dataset


ds = load_dataset("prithivMLmods/Math_symbols")

print(ds.shape)

#this just prints the shape of the dataset 
