import torch
from datasets import load_dataset
# use iam handwritten dataset for training
ds = load_dataset("Teklia/IAM-line")

print(ds)
train = ds["train"]
validation = ds["validation"]
test = ds["test"]

print(train)

x_train, y_train = train["image"], train["label"]
