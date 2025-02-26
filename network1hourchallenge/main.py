import torch
from datasets import load_dataset

ds = load_dataset("prithivMLmods/Math-symbols")

print(ds)
#shape of the dataset:

#changing the pixels into a tensor and refactoring the dataset


#model definition (maybe CNN, research)
class simplecnn(torch.nn.Module):
    def __init__(self):
        super(simplecnn, self).__init__
        self.cnn = torch.nn.Conv2d()
        self.pooling = torch.nn.MaxPool2d()
        self.fcl = torch.nn.Linear()
    def forward(self, x):#
        return x
#training the model

#testing the model with the validation set