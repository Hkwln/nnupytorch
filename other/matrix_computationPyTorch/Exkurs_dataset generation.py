from datasets import Dataset, Features, Value, ClassLabel
# aim of this document is to have a dataset, when printed looks like this:
# it would be also possible to do this on jupiternotebooks
#Dataset({
#    features: ['idx', 'sentence', 'label'],
#    num_rows: 67349
#})
#
#{'idx': Value(dtype='int32', id=None), 'sentence': Value(dtype='string', id=None), 'label': ClassLabel(names=['negative', 'positive'], id=None)}

sentences = ["my dog eats chicken", "this is a positive sentence", "this is a negative sentence"]
#convert labels from str to int
labels = [int(label)for label in["1", "1", "0"]]

features = Features({
    "idx": Value("int32"),
    "sentence": Value("string"),
    "label": ClassLabel(names=["negative", "positiv"])
})
data = {
    "idx": list(range(len(sentences))),
    "sentence": sentences,
    "label": labels
}

dataset = Dataset.from_dict(data, features= features)

print (dataset.features)
#output
#{'idx': Value(dtype='int32', id=None), 'sentence': Value(dtype='string', id=None), 'label': ClassLabel(names=['negative', 'positiv'], id=None)}
print(dataset)