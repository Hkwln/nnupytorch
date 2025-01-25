import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  

class RnnTextClassifier(torch.nn.Module):
    def __init__(self, embeddings, hidden_size= 128, padding_index = -1):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings, freeze= True, padding_idx = padding_index
        )
        self.layer1 = torch.nn.RNN(embeddings.shape[1], hidden_size, batch_first= True)
        self.layer2 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.embedding(x) #.to(device)
        _, h_s = self.layer1(x) # returns(output, hidden_state) 
        # the hidden state captures the last time step, using the output would require additional processing to aggregate the information from all time steps, thus it is not used
        x = torch.relu(h_s[-1])
        x = self.layer2(x)
        return x