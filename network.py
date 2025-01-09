import torch

x = torch.tensor([0,0,1,0], dtype=torch.float) # 4-dim float vector
print(x)

Layer = torch.nn.Linear(4, 5, bias= True) # 4-dim input, 5-dim output -> 5 neurons

z1 = Layer(x)   # forward pass
y1 = torch.sigmoid(z1)
