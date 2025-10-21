import numpy as np
import torch 
# learned positional embeddings
# then sinusodial positional encoding(paper)
# add input to the embeddings, embeddings = embeddings+ positional encoding
class PositionalEncoding(nn.Module):
    __init__(self, seq_len, d):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        #precompute positional encoding once and register as buffer
        p = torch.zeros(seq_len, d)
        position = torch.arange(0, seq_len, 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(-ln(10000)*2*position/d)
        #for even numbers:
        p[:,0::2]= sin(position * div_term)
        #for odd numbers:
        p[:,1::3] = cos(postion * div_term)
        self.register_buffer("p", p)

    def forwar(self):
        pass
