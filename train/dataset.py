import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPImageProcessor

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    )
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

class SceneDataset(Dataset):

    def __init__(self, root_dir, processor, transform=None):

        self.processor = processor
        self.transform = transform
        self.samples = []

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        for cls in self.classes:

            class_dir = os.path.join(root_dir, cls)

            for file in os.listdir(class_dir):

                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):

                    self.samples.append(
                        (
                            os.path.join(class_dir, file),
                            self.class_to_idx[cls]
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
            "path": img_path
        }

def get_dataloader(
    data_dir,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    train=True
):

    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    transform = train_transform if train else test_transform

    dataset = SceneDataset(
        root_dir=data_dir,
        processor=processor,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader, dataset.classes