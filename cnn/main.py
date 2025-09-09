import torch
from datasets import load_dataset
from torchvision import transforms
from model import simplecnn
from datasethandler import MathSymbolsDataset
#               data preperation
ds = load_dataset("prithivMLmods/Math-symbols")
print(torch.cuda.is_available())
#false
print(ds)
train = ds["train"]
print(train)
#shape of the dataset:
x_train,y_train= train["image"], train["label"]

#changing the jpeg into tensor and refactoring the dataset
transform=  transforms.Compose([
    transforms.ToTensor(),              #convert the img to a tensor
    transforms.Normalize((0.5,),(0.5,)) #normalize to mean 0.5 std=0.5
])

train_dataset = MathSymbolsDataset(x_train, y_train, transform)

#training the model
model = simplecnn()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = 1e-4)
losses_train, losses_val = [],[]
accuracy_train, accuracy_val = [],[]

for epoch in 20:
    model.train()
    

#testing the model with the validation set