import torch
#model definition (maybe CNN, research)
class simplecnn(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(simplecnn, self).__init__()
        self.conv = torch.nn.Conv2d(int_channels=3, out_channels=16, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(kernel_size =244,padding= 0 )
        self.fcl = torch.nn.Linear(16, num_classes)
    def forward(self, x):
        x= self.pooling(torch.nn.functional.relu(self.conv(x)))
        x= self.fcl(x)

        return x