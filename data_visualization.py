from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from dataset import TwoClassDataset

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

