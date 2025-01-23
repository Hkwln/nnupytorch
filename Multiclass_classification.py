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
        data, gold_labels = batch
        # forward pass
        predictions = classy(data)
        # loss and backward pass
        loss = loss_func(predictions, gold_labels)
        loss.backward()
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
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
plt.plot(epoch_losses)
plt.show()

test_data = TwoClassDataset()

predicted_class_a = []
predicted_class_b = []
missclassified_class_a = []
missclassified_class_b = []
for i in range(len(test_data)):
    x, label = test_data[i]
    pred = classy(x)
    probs = torch.softmax(pred, dim=0) # note that the softmax is mathematically not needed since it is a monotonic function
    class_pred = torch.argmax(probs, dim=0)
  
    if label != class_pred:
        if label == 0:
            missclassified_class_a.append(x.unsqueeze(dim=0))
        else:
            missclassified_class_b.append(x.unsqueeze(dim=0))
    else:
        if class_pred == 0:
            predicted_class_a.append(x.unsqueeze(dim=0))
        else:
            predicted_class_b.append(x.unsqueeze(dim=0))
predicted_class_a = torch.cat(predicted_class_a, dim=0) # concatenates tensor along specified, existing dimension, does not create new dimension
predicted_class_b = torch.cat(predicted_class_b, dim=0) # `stack` is an alternative which creates a new dimension (i.e. we wouldn't have `unsqueeze`d the vectors)
missclassified_class_a = torch.cat(missclassified_class_a, dim=0)
missclassified_class_b = torch.cat(missclassified_class_b, dim=0)

fig, ax = plt.subplots()
ax.scatter(predicted_class_a[:,0], predicted_class_a[:,1], label="class 1", c="darkorange")
ax.scatter(predicted_class_b[:,0], predicted_class_b[:,1], label="class 2", c="dodgerblue")
ax.scatter(missclassified_class_a[:,0], missclassified_class_a[:,1], c="darkorange", edgecolors="r")
ax.scatter(missclassified_class_b[:,0], missclassified_class_b[:,1], c="dodgerblue", edgecolors="r")
ax.legend()
plt.show()
