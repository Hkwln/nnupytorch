import zipfile
from functools import partial
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import trange