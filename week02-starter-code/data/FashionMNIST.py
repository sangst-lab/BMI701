from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


classes_fashion_mnist = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


folder_map = {
    "train": [
        ("Train Images", "Train Labels"),
        ("train images", "train labels"),
    ],
    "test": [
        ("T10K Images", "T10K Labels"),
        ("test images", "test labels"),
        ("Test Images", "Test Labels"),
    ],
}


class FashionMNISTDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, return_path=False):
        if split not in folder_map:
            raise ValueError("split must be 'train' or 'test'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_dir, self.label_dir = self._find_split_folders(split)
        self.transform = transform
        self.return_path = return_path

        self.image_paths = sorted(self.image_dir.glob("*.png"))


    def _find_split_folders(self, split):
        for image_folder, label_folder in folder_map[split]:
            image_dir = self.root_dir / image_folder
            label_dir = self.root_dir / label_folder
            if image_dir.exists() and label_dir.exists():
                return image_dir, label_dir
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("L")
        label = int(label_path.read_text(encoding="utf-8").splitlines()[0])

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)

        return image, label


def get_transform():
    return transforms.ToTensor()


def get_dataloaders(root_dir, batch_size=64, validation_size=5000, num_workers=0):

    transform = get_transform()

    full_train_dataset = FashionMNISTDataset(
        root_dir=root_dir,
        split="train",
        transform=transform,
    )
    test_dataset = FashionMNISTDataset(
        root_dir=root_dir,
        split="test",
        transform=transform,
    )

    validation_size=5000
    train_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    dataset = FashionMNISTDataset(
        root_dir=r"E:\data\BMI701\FashionMNIST",
        split="train",
        transform=transforms.ToTensor()
    )

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    for i, ax in enumerate(axes.flatten()):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"{i+1:06d}: {classes_fashion_mnist[label]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
