from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

class TwoClassDataset(Dataset):
    def __init__(self):
        # create synthetic dataset
        self.features_a = torch.normal(mean=1., std=1.0, size=(100,2))
        self.labels_a = torch.zeros(100, dtype=torch.long)
        self.features_b = torch.normal(mean=-1.0, std=1.0, size=(100,2))
        self.labels_b = torch.ones(100, dtype=torch.long)

        self.data = torch.cat((self.features_a, self.features_b), dim=0) # X: 200 x 2 matrix : [x1,x2]
        self.labels = torch.cat((self.labels_a, self.labels_b), dim=0) # Y: binary vector of length 200: 0: class A, 1: class B
    
    def __len__(self):
        return self.data.size(dim=0)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
train_data = TwoClassDataset()
#split dataset
split = int(0.2 * len(train_data))
indices = list(range(len(train_data)))
training_data, test_data = indices[split:], indices[:split]
train_subset = Subset(train_data, training_data)
test_subset = Subset(train_data, test_data)

# create a DataLoader
train_dataloader = DataLoader(train_subset, batch_size = 5 , shuffle = True)
test_dataset = DataLoader(test_subset, batch_size = 5 , shuffle = True)


# plot both classes
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
plt.scatter(train_features[:,0], train_features[:,1], c=train_labels, cmap='viridis')
plt.xlabel(' Feature 1')
plt.ylabel(' Feature 2')
plt.title('Two Class Data')
plt.show()

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define the layers
        return 0
            
    def forward(self, x):
        # define the forward pass
        return x
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
        
        # access data and labels from batch

        # forward pass
       
        # loss and backward pass
        
        # update network weights
        
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
