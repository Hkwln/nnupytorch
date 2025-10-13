import torch
from datasets import load_dataset
from torchvision import transforms
from model import simplecnn
from datasethandler import MathSymbolsDataset
from torch.utils.data import DataLoader
#               data preperation
ds = load_dataset("prithivMLmods/Math-symbols")
# the dataset has a train vaidation and test batch
validation =ds["validation"]
train = ds["train"]
#shape of the dataset:
x_train,y_train= train["image"], train["label"]
x_test ,y_test = validation["image"], validation["label"]

#changing the jpeg into tensor and refactoring the dataset
transform=  transforms.Compose([
    transforms.ToTensor(),              #convert the img to a tensor
    transforms.Normalize((0.5,),(0.5,)) #normalize to mean 0.5 std=0.5
])

train_dataset = MathSymbolsDataset(x_train, y_train, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


model = simplecnn()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = 1e-4)
losses_train, losses_val = [],[]
accuracy_train, accuracy_val = [],[]
num_epochs = 2
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for image, labels in train_loader:
        outputs = model(image)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

#testing the model with the validation set
# todo: there is model.eval(), does that work that way? also 
validation_dataset = MathSymbolsDataset(x_test, y_test, transform)
model.eval()
count = 0
with torch.no_grad():
    
    for i in range(len(validation_dataset)):
    
        for image, label in validation_dataset:
            outputs = model(image).unsqueeze(0) # adding a batch dim with the unsqueeze
            _, predicted = torch.max(outputs.data,1)
            if predicted == label:
                count= count +1
print(f"your model got {count} samples out of {len(validation_dataset) correct ")
