#Given y = (Wx +b)^T (Wx+b)
import torch

W = torch.tensor([[1.0,2.0],[3.0,4.0]],requires_grad= True)
x = torch.tensor([[1.0],[2.0]],requires_grad= True)
b = torch.tensor([[1.0],[2.0]],requires_grad= True)

Wx_b = torch.matmul(W,x) + b


#transpose
y = torch.matmul(Wx_b.t(), Wx_b)

print(y)

#Calculate the gradients of y with respect to W and b
y.backward()

