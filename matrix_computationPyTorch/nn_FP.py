import torch

class SimpleForwardPass(torch.nn.Module)
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 3)
        self.layer2 = torch.nn.Linear(3, 4)
        self.layer3 = torch.nn.Linear(4,2)
        self.relu = torch.nn.ReLu()
        
        #defining custom parameters for the model
        self.param_w_1 = torch.nn.Parameter([0.5, -0.2, 0.5, 0.4 ],[-0.3, 0.8,-0.5,0.2],[0.2,-0.1,0.3,-0.4])
        self.param_b_1 = torch.nn.Parameter([],[],[])
        self.param_w_2 = torch.nn.Parameter( )
        self.param_b_2 = torch.nn.Parameter()

    def forward(self, x):

        return x