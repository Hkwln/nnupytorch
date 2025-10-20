import torch
from datasets import load_dataset
from torchvision import transforms
from model import simplecnn
from datasethandler import MathSymbolsDataset
from torch.utils.data import DataLoader
import os
from tensorboard import SummaryWriter
current_dire = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dire,"model.pt")
#               data preperation
ds = load_dataset("prithivMLmods/Math-symbols")
# the dataset has a train vaidation and test batch
validation =ds["validation"]
train = ds["train"]
#shape of the dataset:
x_train,y_train= train["image"], train["label"]

x_val, y_val= validation["image"], validation["label"]
x_test ,y_test = train["image"], train["label"]

#changing the jpeg into tensor and refactoring the dataset
transform=  transforms.Compose([
    transforms.ToTensor(),              #convert the img to a tensor
    transforms.Normalize((0.5,),(0.5,)) #normalize to mean 0.5 std=0.5
])

train_dataset = MathSymbolsDataset(x_train, y_train, transform)
validation_dataset = MathSymbolsDataset(x_val, y_val, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader= DataLoader(validation_dataset, batch_size=32, shuffle=True)

writer =SummaryWriter('runs/cnn')
model = simplecnn()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = 1e-4)
losses_train, losses_val = [],[]
accuracy_train, accuracy_val = [],[]
num_epochs = 2
counter: int = 0
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for image, labels in train_loader:
        counter +=1
        outputs = model(image)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss, counter)
    #after a epoch, evaluate my model on the validation_dataset
    #use torch.eval() and torch.no_grad()
    # Calculate the validatioin loss and accuracy
    model.eval() # go into validation mode
    val_loss = 0
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for image, labels in val_loader:
            out = model(image)
            counter +=1
            
            writer.add_scalar('validation', val_accuracy, counter)
            writer.add_scalar('validation loss', val_loss, counter)
            loss = loss_fn(out, labels)
            val_loss = loss.item() *labels.size(0)
            _, predictions = torch.max(out, 1)
            #count the correct predictions
            #if predictions == label:correct=correct+1
            correct += (predictions == labels).sum().item() 
            #update total number of samples
            total += labels.size(0)

    # loss per samples
    val_loss = val_loss/total

    #total correct/total samples = accuracy
    val_accuracy = correct/total
    print(f"validation lost: {val_loss}, validation accuracy={val_accuracy}")
    #logging validation metrics, val accuracy, val loss
    model.train()# go back into training mode
    # every few epochs store params in histogram:
    for name, param in model.parameters():
        writer.add_histogram('model_state', param.data().cpu().numpy(), epoch)
writer.close()
    
#safe the the trained model inside a .pt format
current_state = {
        "model_state dic": model.state_dict(),
        "optimizer state dict" : optimizer.state_dict(),
        "loss" : loss,
        }  
torch.save(current_state, path )
#testing the codel with the validation set
# todo: there is model.eval(), does that work that way? also 
train_dataset = MathSymbolsDataset(x_test, y_test, transform)
model.eval()
count: int = 0
with torch.no_grad():
        for image, label in train_dataset:
            #add batch dimension
            image = image.unsqueeze(0)
            outputs = model(image)
            _, predicted = torch.max(outputs.data,1)
            if predicted == label:
                count = count+1
print(f"your model got {count} samples out of {len(train_dataset)} correct ")
