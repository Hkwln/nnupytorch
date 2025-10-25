import torch
#here is the cnn model
# todo: do some research what kind of cnn structure is recomendet
# maybe it is time to do a residual neural network

class cnn(torch.nn.Modul):
    def __init__(self):

        super(cnn, self).__init__()
        self.convolutionlayer = torch.nn.Conv2d(in_channels=3, out_channels= all_chars, kernel_size=16)
        self.pooling = torch.nn.Maxpool2d(kernel_size=3, stride=2)
        self.fc = torch.nn.Linear(num_after_convolution,num_classes)
    def forward(x),
        b= self.pooling(torch.nn.functional.relu(self.convolutionlayer(x)))
        b = self.fc(b)
        out = b+x
        return out



