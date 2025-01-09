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
        self.layer3 = torch.sigmoid(torch.nn.Linear(3, 1))
        
    def forward(self, x):
        # define the forward pass
        y1 = torch.relu(self.layer1(x))
        y2 = torch.relu(self.layer2(y1))
        y3 = torch.sigmoid(self.layer3(y2))
        
        return y3
    
#weiter bei aufgabe 7