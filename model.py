import torch
from PositionalEncodingLayer import PositionalEncoding
#todo: finish
class Transformermodel(torch.nn.Module):

    def __init_(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super(Transformermodel).__init__()
        
        self.transformer = torch.nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers= num_decoder,dropout=dropout_p)
        def forward(self):
            pass
