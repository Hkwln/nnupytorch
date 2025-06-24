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
        self.pooling = torch.nn.MaxPool2d(kernel_size =244,padding= 0 )
        self.fcl = torch.nn.Linear()
    def forward(self, x):
        return x
#training the model
model = simplecnn()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = 1e-4)
losses_train, losses_val = [],[]
accuracy_train, accuracy_val = [],[]

for epoch in 20:
    model.train()
    

#testing the model with the validation set