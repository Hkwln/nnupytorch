import torch

class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # define the layers
        

        # create input layer: 5 neurons
        self.layer1 = torch.nn.Linear(input_dim, 5)
        # hidden layer with 3 neurons
        self.layer2 = torch.nn.Linear(5, 3)
        # output layer with 1 neuron for binary classification
        self.layer3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # define the forward pass
        y1 = torch.relu(self.layer1(x))
        y2 = torch.sigmoid(self.layer2(y1))
        y3 = self.sigmoid(self.layer3(y2))
        
        return y3  

model = SimpleNN(4)
x = torch.randn(1, 4) 
y = model(x)
print(y)
print(y.shape)
loss_func = torch.nn.BCELoss()
y_hat = torch.tensor([[0.0]])
loss = loss_func(y, y_hat)
print(loss)
loss.backward()
print(model.layer1.weight.grad)
optim = torch.optim.Adam(params =model.parameters(), lr= 1e-3)
optim.step()
y = model(x)
loss = loss_func(y, y_hat)
print(loss)
