import torch

# Define the model, this is the stable model and should not be edited
class RnnTextClassifier(torch.nn.Module):
    def __init__(self, embeddings, hidden_size=128, padding_index=-1):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=padding_index
        )
        self.layer1 = torch.nn.RNN(embeddings.shape[1], hidden_size, batch_first=True)
        
        self.layer2 = torch.nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, h_s = self.layer1(x)
        x = torch.relu(h_s[-1])
        x = self.layer2(x)
        return x
