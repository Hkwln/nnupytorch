from imports import *
from rnn import RnnTextClassifier

sst2 = load_dataset("stanfordnlp/sst2")
#dataset structure:
# three splits: train, validation, test
# each of these has fetures: ['idx', 'sentence', 'label']
# and the number of the rows num_rows
dataset_train = sst2["train"]
# Download the GloVe embeddings
glove = hf_hub_download("stanfordnlp/glove", "glove.6B.zip")

with zipfile.ZipFile(glove, "r") as f:
    print(f.namelist())

#There are multiple files with differnt dimensionality ofthe features in the zip archive: 50d,100d,200d,300d
filename = "glove.6B.300d.txt"
with zipfile.ZipFile(glove, "r") as f:
    for idx, line in enumerate(f.open(filename)):
        print(line)
        if idx == 5:
            break

# Unpack the downloaded file
word_to_index = dict()
embeddings = []

with zipfile.ZipFile(glove, "r") as f:
    for idx, line in enumerate(f.open(filename)):
        values = line.split()
        word = values[0].decode("utf-8")
        features = torch.tensor([float(value) for value in values[1:]])
        word_to_index[word] = idx
        embeddings.append(features)

# Last token in the vocabulary is '<unk>' which is used for out-of-vocabulary words
# We also add a '<pad>' token to the vocabulary for padding sequences
word_to_index["<pad>"] = len(word_to_index)
padding_token_id = word_to_index["<pad>"]
unk_token_id = word_to_index["<unk>"]

embeddings.append(torch.zeros(embeddings[0].shape))

# Convert the list of tensors to a single tensor
embeddings = torch.stack(embeddings)
print(f"Embedding shape: {embeddings.size(1)}")

def tokenize(text: str):
    return text.lower().split()


def map_token_to_index(token):
    # Return the index of the token or the index of the '<unk>' token if the token is not in the vocabulary
    return word_to_index.get(token, unk_token_id)


def map_text_to_indices(text: str):
    return [map_token_to_index(token) for token in tokenize(text)]


def prepare_dataset(dataset):
    return dataset.map(lambda x: {"token_ids": map_text_to_indices(x["sentence"])})


dataset_train_tokenized = prepare_dataset(dataset_train)


def pad_inputs(batch, keys_to_pad=["token_ids"], padding_value=-1):
# Pad keys_to_pad to the maximum length in batch
    padded_batch = {}
    for key in keys_to_pad:
        # Get maximum length in batch
        max_len = max([len(sample[key]) for sample in batch])
        # Pad all samples to the maximum length
        padded_batch[key] = torch.tensor(
            [
                sample[key] + [padding_value] * (max_len - len(sample[key]))
                for sample in batch
            ]
        )
    # Add remaining keys to the batch
    for key in batch[0].keys():
        if key not in keys_to_pad:
            padded_batch[key] = torch.tensor([sample[key] for sample in batch])
    return padded_batch

def get_dataloader(dataset, batch_size=32, shuffle=False):
    # Create a DataLoader for the dataset
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(pad_inputs, padding_value=padding_token_id),
        shuffle=shuffle,
    )


# We select the columns that we want to keep in the dataset
dataset_train_tokenized = dataset_train_tokenized.with_format(
    columns=["token_ids", "label"]
)
# Create a DataLoader for the training dataset
dataloader_train = get_dataloader(dataset_train_tokenized, batch_size=8, shuffle=True)

for batch in dataloader_train:
    token_ids = batch["token_ids"]
    labels = batch["label"]
    break

#here is a simpletextclassifier
class SimpleTextClassifier(torch.nn.Module):
    def __init__(self, embeddings, hidden_size=128, padding_index=-1):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=padding_index
        )
        self.layer1 = torch.nn.Linear(embeddings.shape[1], hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        # By summing the embeddings of all tokens in the sequence, we get a bag-of-words vector for each sample input
        x = torch.sum(x, dim=1)
        x = torch.relu(self.layer1(x))
        x = self.output_layer(x)
        return x
#we can now feed the model into the SimpleTextClassifier, or the Rnntextclassifier
model1 = SimpleTextClassifier(embeddings, padding_index=padding_token_id)
model2 = RnnTextClassifier(embeddings,padding_index=padding_token_id)
print("simpleTextClassifier\n")
print(model1(torch.tensor(dataset_train_tokenized["token_ids"][:2])))
print("RnnTextClassifier\n")
print(model2(torch.tensor(dataset_train_tokenized["token_ids"][:2])))

#Evaluation
def compute_accuracy(predictions: torch.tensor, labels: torch.tensor):
    return torch.sum(torch.argmax(predictions, dim=1) == labels).item() / len(labels)


def evaluate_model(model, dataset, loss_fn=None):
    # Compute the accuracy and optionally the loss of the model on the dataset
    dataloader = get_dataloader(dataset, batch_size=32)
    accuracies = []
    losses = []
    # We don't need to compute gradients for the evaluation
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

dataset_val = sst2["validation"]
dataset_val_tokenized = prepare_dataset(dataset_val)
dataset_val_tokenized = dataset_val_tokenized.with_format(
    columns=["token_ids", "label"]
)
accuracy, _ = evaluate_model(model1, dataset_val_tokenized)
print(f"Accuracy on the validation dataset: {accuracy}")
#Training


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model1.parameter(), lr = 1e-3)
optimizer.step()
losses_train, losses_val = []
accuracies_train, accuracies_val = []

#compute loss and accuracy on the training model
accuracy, loss = evaluate_model(model1, dataset_val_tokenized, loss_fn)
losses_train.append(loss)
accuracies_train.append(accuracy)

#compute loss and accuracy on the validation set

accuracy, loss = evaluate_model(model1, dataset_val_tokenized, loss_fn)
losses_val.append(loss)
accuracies_val.append(accuracy)

NUM_EPOCHS = 10

# A progress bar to visualize the training progress
pbar = trange(NUM_EPOCHS)
#Training Loop
for epoch in pbar:
    #Do one epoch of training
    for batch in dataloader_train:
        token_ids = batch["token_ids"]
        labels = batch["label"]

        #forward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate the loss and accuracy on the training set
        acc_train, loss_train = evaluate_model(model2, dataset_train_tokenized, loss_fn)
        accuracies_train.append(acc_train)
        losses_train.append(loss_train)

        # Evaluate the model on the validation set
        acc_val, loss_val = evaluate_model(model1, dataset_val_tokenized, loss_fn)
        accuracies_val.append(acc_val)
        losses_val.append(loss_val)

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