import torch

#this is a planning model which is not currently in use, because the same model is in SA.py
#this model is for experimenting purposes, 
#in case you want to use this, just import it in SA.py and type model2 = ExperimentalTextClassifier

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  

class ExperimentalTextClassifier(torch.nn.Module):
    def __init__(self, embeddings, hidden_size= 128, padding_index = -1):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings, freeze= True, padding_idx = padding_index
        )
        self.layer1 = torch.nn.GRU(embeddings.shape[1], hidden_size, batch_first= True)
        self.dropout = torch.nn.Dropout(0.5)
        self.layer2 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.embedding(x) #.to(device)
        _, h_s = self.layer1(x) # returns(output, hidden_state) 
        # the hidden state captures the last time step, using the output would require additional processing to aggregate the information from all time steps, thus it is not used
        x = torch.relu(h_s[-1])
        x = self.dropout(x)
        x = self.layer2(x)
        return x