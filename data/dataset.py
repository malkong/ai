# train/dataset.py

from torch.utils.data import Dataset
from PIL import Image


class AACDataset(Dataset):

    def __init__(self, image_paths, labels, preprocess=None):

        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.preprocess:
            image = self.preprocess(image)

        label = self.labels[idx]

        return image, label