from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

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

training_data = (train_data[:,0], train_data.features_a[:,1])
test_data = (train_data.features_b[:,0], train_data.features_b[:,1])

# create a DataLoader
train_dataloader = DataLoader(training_data, batch_size = 5 , shuffle = True)
test_data = DataLoader(test_data, batch_size = 5 , shuffle = True)



# plot both classes
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap = 'gray')
plt.show()
print(f"Label: {label}")
