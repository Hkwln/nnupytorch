import numpy as np
import torch 
import math
# learned positional embeddings
# then sinusodial positional encoding(paper)
# add input to the embeddings, embeddings = embeddings+ positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d, drop_prob: float=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        self.base = float(base)
        self.batch_first = batch_first
        self.dropout = torch.nn.Dropout(drop_prob)
        #precompute positional encoding once and register as buffer
        pe = torch.zeros(seq_len, d)
        position = torch.arange(0, seq_len, 1, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(0, d,dtype=torch.float32)
        div_term = torch.exp(-torch.log(10000)*2*dim/d)
        #for even numbers:
        pe[:,0::2]= torch.sin(position * div_term)
        #for odd numbers:
        pe[:,1::2] = torch.cos(postion * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
            # expected shape: (batch, seq_len, d) when batch_first=True
        if x.dim() != 3:
            raise ValueError(f"Expected input with 3 dims (batch, seq, d); got {x.dim()}")
        batch_size, seq_len, d = x.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: input d={d}, module d_model={self.d_model}")
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

    # get slice and move to input dtype/device
        pe_slice = self.pe[:seq_len].to(dtype=x.dtype, device=x.device)  # (seq_len, d)
        pe_slice = pe_slice.unsqueeze(0)  # (1, seq_len, d)
        x = x + pe_slice  # broadcast over batch
        return self.dropout(x)
