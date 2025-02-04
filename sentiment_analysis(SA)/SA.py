from imports import *
#from rnn import RnnTextClassifier


#here you can change the device to work, not yet working
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using device: {device}')

#here is the path for the preprocessed data
path = "preprocessed_data.pt"
#here is the path for the saved model
path2 = "saved_model.pt"
load = False
#save embeddings and tokenized dataset to disk
def save_preprocessed(path):
    preprocessed_data ={
        "embeddings":embeddings,
        "dataset_train_tokenized": dataset_train_tokenized,
        "dataset_val_tokenized": dataset_val_tokenized
    }
    #save the embeddings tensor
    torch.save(preprocessed_data, path)
  

def load_preprocessed(path):
    data = torch.load(path, weights_only = True)
    return data["embeddings"],data["dataset_train_tokenized"], data["dataset_val_tokenized"]

if os.path.exists(path):
    embeddings, dataset_train_tokenized, dataset_val_tokenized = load_preprocessed(path)
    Load = True


if not load:
    sst2 = load_dataset("stanfordnlp/sst2")
    #dataset structure:
    # three splits: train, validation, test
    # each of these has features: ['idx', 'sentence', 'label']
    # and the number of the rows num_rows
    dataset_train = sst2["train"]
    dataset_val = sst2["validation"]

    # Download the GloVe embeddings
    glove = hf_hub_download("stanfordnlp/glove", "glove.6B.zip")

    # There are multiple files with different dimensionality of the features in the zip archive: 50d, 100d, 200d, 300d
    filename = "glove.6B.300d.txt"

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

    embeddings.append(torch.zeros(embeddings[0].shape)) # Add <pad> token embedding
    embeddings.append(torch.zeros(embeddings[0].shape)) # Add <unk> token embedding

    # Convert the list of tensors to a single tensor
    embeddings = torch.stack(embeddings)

# Tokenize the sentences
def tokenize(text: str):
    return text.lower().split()

def map_token_to_index(token):
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
if not load:
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
if not load:
    train_dataloader = get_dataloader(dataset_train_tokenized, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(dataset_val_tokenized, batch_size=32, shuffle=False)
    save_preprocessed(path)


def save_model(path):
    current_state = {
        "model_state_dict": model2.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss
    }
    torch.save(current_state, path)
def load_model(path):
    data =torch.load(path, weights_only= True)
    return data["model_state_dict"],data["optimizer_state_dict"], data["loss"]

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

# Instantiate the model, if you instead want to use the experimental model, do it here
model2 = RnnTextClassifier(embeddings, padding_index=padding_token_id)
#load the last model state if available
if os.path.exists(path2):
    model2_state_dict, optimizer, loss = load_model(path2)
    model2.load_state_dict(model2_state_dict)


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

# Training loop with adjusted learning rate
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model2.parameters(), lr=1e-4)  # Adjusted learning rate
losses_train, losses_val = [], []
accuracies_train, accuracies_val = [], []

NUM_EPOCHS = 2
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
    train_accuracy, train_loss = evaluate_model(model2, train_dataloader, loss_fn)
    losses_train.append(train_loss)
    accuracies_train.append(train_accuracy)
    
    # Compute loss and accuracy on the validation set
    val_accuracy, val_loss = evaluate_model(model2, val_dataloader, loss_fn)
    losses_val.append(val_loss)
    accuracies_val.append(val_accuracy)

    pbar.set_postfix_str(
        f"Train loss: {losses_train[-1]} - Validation acc: {accuracies_val[-1]}"
    )
#Saving the model weights and biases
save_model(path2)

# Visualize the loss and accuracy
plt.plot(losses_train, color="orange", linestyle="-", label="Train loss")
plt.plot(losses_val, color="orange", linestyle="--", label="Validation loss")
plt.plot(accuracies_train, color="steelblue", linestyle="-", label="Train accuracy")
plt.plot(accuracies_val, color="steelblue", linestyle="--", label="Validation accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()