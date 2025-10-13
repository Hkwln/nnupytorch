import torch

#model definition (maybe CNN, research)
class simplecnn(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(simplecnn, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcl = torch.nn.Linear(16 * 111 * 111, num_classes)
    def forward(self, x):
        #ensure that if just a single image is passed instead of a barch, a dimension is added
        if x.dim() == 3:
            x =x.unsqueeze(0)
        x = self.pooling(torch.nn.functional.relu(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten for the linear layer
        x = self.fcl(x)
        return x
