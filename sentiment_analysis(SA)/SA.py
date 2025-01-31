from imports import *
from rnn import RnnTextClassifier
# here you can change the device to work on gpu
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using device: {device}')

sst2 = load_dataset("stanfordnlp/sst2")
#dataset structure:
# three splits: train, validation, test
# each of these has features: ['idx', 'sentence', 'label']
# and the number of the rows num_rows
dataset_train = sst2["train"]
dataset_val = sst2["validation"]

# Download the GloVe embeddings
glove = hf_hub_download("stanfordnlp/glove", "glove.6B.zip")

#with zipfile.ZipFile(glove, "r") as f:
#    print(f.namelist())

# There are multiple files with different dimensionality of the features in the zip archive: 50d, 100d, 200d, 300d
filename = "glove.6B.300d.txt"
#with zipfile.ZipFile(glove, "r") as f:
#    for idx, line in enumerate(f.open(filename)):
#        #print(line)
#        if idx == 5:
#            break

# Unpack the downloaded file
word_to_index = dict()
embeddings = []

with zipfile.ZipFile(glove, "r") as f:
    with f.open(filename) as file:
        for idx, line in enumerate(file):
            values = line.split()
            word = values[0].decode("utf-8")
            features = torch.tensor([float(value) for value in values[1:]], dtype=torch.float32)
            word_to_index[word] = idx
            embeddings.append(features)

# Last token in the vocabulary is '<unk>' which is used for out-of-vocabulary words
# We also add a '<pad>' token to the vocabulary for padding sequences
word_to_index["<pad>"] = len(word_to_index)
padding_token_id = word_to_index["<pad>"]
word_to_index["<unk>"] = len(word_to_index)
unk_token_id = word_to_index["<unk>"]

embeddings.append(torch.zeros(embeddings[0].shape))  # Add <pad> token embedding
embeddings.append(torch.zeros(embeddings[0].shape))  # Add <unk> token embedding

# Convert the list of tensors to a single tensor
embeddings = torch.stack(embeddings)

# Tokenize the sentences
def tokenize(text: str):
    return text.lower().split()

def map_token_to_index(token):
    # Return the index of the token or the index of the '<unk>' token if the token is not in the vocabulary
    return word_to_index.get(token, unk_token_id)

def map_text_to_indices(text: str):
    tokens = tokenize(text)
    return [map_token_to_index(token) for token in tokens]

# Tokenize and map the dataset
def prepare_dataset(dataset):
    tokenized_dataset = []
    for example in dataset:
        token_ids = map_text_to_indices(example["sentence"])
        tokenized_dataset.append({"token_ids": token_ids, "label": example["label"]})
    return tokenized_dataset

dataset_train_tokenized = prepare_dataset(dataset_train)
dataset_val_tokenized = prepare_dataset(dataset_val)

# Create a DataLoader
def pad_inputs(batch, keys_to_pad=["token_ids"], padding_value=padding_token_id):
    padded_batch = {}
    for key in keys_to_pad:
        max_len = max([len(sample[key]) for sample in batch])
        padded_batch[key] = torch.tensor(
            [
                sample[key] + [padding_value] * (max_len - len(sample[key]))
                for sample in batch
            ],
            dtype=torch.long  # Ensure the tensor is of type long
        )
    for key in batch[0].keys():
        if key not in keys_to_pad:
            padded_batch[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.long)
    return padded_batch

def get_dataloader(dataset, batch_size=32, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(pad_inputs, padding_value=padding_token_id),
        shuffle=shuffle,
    )

train_dataloader = get_dataloader(dataset_train_tokenized, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(dataset_val_tokenized, batch_size=32, shuffle=False)

# Define the model
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

# Instantiate the model
model2 = RnnTextClassifier(embeddings, padding_index=padding_token_id)

# Define the evaluation function
def evaluate_model(model, dataloader, loss_fn=None):
    accuracies = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch["token_ids"]
            labels = batch["label"]
            predictions = model(token_ids)
            if loss_fn:
                loss = loss_fn(predictions, labels)
                losses.append(loss.item())
            accuracies.append(compute_accuracy(predictions, labels))
    return sum(accuracies) / len(accuracies), (
        (sum(losses) / len(losses)) if loss_fn else None
    )

# Define the compute_accuracy function
def compute_accuracy(predictions: torch.tensor, labels: torch.tensor):
    return torch.sum(torch.argmax(predictions, dim=1) == labels).item() / len(labels)

# Training loop
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model2.parameters(), lr=1e-3)
losses_train, losses_val = [], []
accuracies_train, accuracies_val = [], []

NUM_EPOCHS = 10
pbar = trange(NUM_EPOCHS)

for epoch in pbar:
    model2.train()
    for batch in train_dataloader:
        token_ids = batch["token_ids"]
        labels = batch["label"]

        predictions = model2(token_ids)

        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute loss and accuracy on the training set
    accuracy, loss = evaluate_model(model2, train_dataloader, loss_fn)
    losses_train.append(loss)
    accuracies_train.append(accuracy)
    
    # Compute loss and accuracy on the validation set
    accuracy, loss = evaluate_model(model2, val_dataloader, loss_fn)
    losses_val.append(loss)
    accuracies_val.append(accuracy)

    pbar.set_postfix_str(
        f"Train loss: {losses_train[-1]} - Validation acc: {accuracies_val[-1]}"
    )

# Visualize the loss and accuracy
plt.plot(losses_train, color="orange", linestyle="-", label="Train loss")
plt.plot(losses_val, color="orange", linestyle="--", label="Validation loss")
plt.plot(accuracies_train, color="steelblue", linestyle="-", label="Train accuracy")
plt.plot(accuracies_val, color="steelblue", linestyle="--", label="Validation accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()