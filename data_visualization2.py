from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import *

train_data = TwoClassDataset()

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define the layers
        self.layer1 = torch.nn.Linear(2,5)
        self.layer2 = torch.nn.Linear(5,2)
            
    def forward(self, x):
        # define the forward pass
        y1 = torch.relu(self.layer1(x))
        y2 = self.layer2(y1)
        return y2
# instantiate model, loss criterion and optimizer
classy = Classifier()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=classy.parameters(), lr=0.01)

# store epoch metrics
epoch_accs = []
epoch_losses = []
dataloader = DataLoader(dataset=train_data, batch_size=25, shuffle=True)

# epoch loop
for epoch in tqdm(range(50)):
    epoch_acc = []
    epoch_loss = []
    
    # mini-batch loop for one epoch
    for batch in dataloader:
        # reset gradients to 0
        optim.zero_grad()
        # access data and labels from batch
        data, gold_labels = next(iter(dataloader))
        # forward pass
        predictions = classy(data)
        # loss and backward pass
        loss = loss_func(predictions, gold_labels)
        # update network weights
        optim.step()
        # check accuracy (get predicted class for each sample, compare to gold label)
        category_probs = torch.softmax(predictions, dim=1) # sums up to 1 for each sample
        category_labels = torch.argmax(category_probs, dim=1) # extract most likely label
        batch_acc = (category_labels == gold_labels).float().sum(dim=0)/25.0 # avg accuracy for batch
        epoch_acc.append(batch_acc.item())
        epoch_loss.append(loss.item())

    # average all metrics across one epoch
    epoch_losses.append(sum(epoch_loss)/len(epoch_loss))
    epoch_accs.append(sum(epoch_acc)/len(epoch_acc))

plt.plot(epoch_accs)
plt.plot(epoch_losses)

