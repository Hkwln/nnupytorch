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
