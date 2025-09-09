import torch
#model definition (maybe CNN, research)
class simplecnn(torch.nn.Module):
    def __init__(self):
        super(simplecnn, self).__init__
        self.cnn = torch.nn.Conv2d()
        self.pooling = torch.nn.MaxPool2d(kernel_size =244,padding= 0 )
        self.fcl = torch.nn.Linear()
    def forward(self, x):
        return x