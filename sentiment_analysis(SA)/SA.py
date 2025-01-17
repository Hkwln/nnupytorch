from imports import *

sst2 = load_dataset("stanfordnlp/sst2")
#dataset structure:
# three splits: train, validation, test
# each of these has fetures: ['idx', 'sentence', 'label']
# and the number of the rows num_rows

dataset_train = sst2['train']
print(dataset_train[0])

glove = hf_hub_download("stanfort/glove", "glove.6B.zip")
with zipfile.ZipFile(glove, 'r') as f:
    print(f.namelist())

#There are multiple files with differnt dimensionality ofthe features in the zip archive: 50d,100d,200d,300d
filename = "glove.6B.300d.txt"
with zipfile.ZipFile(glove, "r") as f:
    for idx, line in enumerate(f.open(filename)):
        print(line)
        if idx == 5;
            break
# unpack the downloaded file
word_to_index = dict{}
embeddings = []

with zipfile.ZipFile(glove, "r") ad f:
    for idx, line in enumerate(f.open(filename)):
        values = line.split()
        word = values[0].decode('utf-8')
        features = torch.tensor([float(value)for value in values[1:]])
        word_to_index[word] = idx
        embeddings.append(features)
