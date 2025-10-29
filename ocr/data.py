from datasets import load_dataset
from torchvision import transforms
# use iam handwritten dataset for training
ds = load_dataset("Teklia/IAM-line")
#Note that all images are resized to a fixed height of 128 pixels.
print(ds)
train = ds["train"]
validation = ds["validation"]
test = ds["test"]

print(train)

x_train, y_train = train["image"], train["label"]


transfomr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
