import torch
from PositionalEncodingLayer import PositionalEncoding
#todo: finish
#layers of a transformer models:
# input embedding layer, 
# PositionalEncoding,
# Multi-head self-Attention Mechanism:
#   for each token, compute attention scores with every other token
#   each head learns different types of relationships
#   nn.MultiheadAttention(embed_dim, num_heads)
# Feed-Forward Neural Networks
# Normalization and residual connections
# nn.LayerNorm(embed_dim), out= LayerNorm(x+Sublayer(x))
# linear layer relu linear layer followed redidual 
# output layer
class Transformermodel(torch.nn.Module):

    def __init_(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super(Transformermodel).__init__()
        
        self.transformer = torch.nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers= num_decoder,dropout=dropout_p)
        self.embedding = torch.nn.Embedding (vocab_size, embed_dim)
        self.positionalEncoding = PositionalEncoding()
        self.linear = torch.nn.Linear(embed_dim, vocab_size)
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            #embedding layer with a result of vectors shape:[batch_size, seq_lenm embedding_dim]
            src = self.embedding(src) *math.sqrt(self.embedding.embedding_dim)
            src = self.positionalEncoding(src)
            tgt = self.embedding(src) * math.sqrt(self.embedding.embed_dim)
            tgt = self.positionalEncoding(tgt)
            out = self.transformer(src, tgt, src_mask= src_mast, tgt_mask=tgt_mask)
            out = self.linear(out)

            return out
