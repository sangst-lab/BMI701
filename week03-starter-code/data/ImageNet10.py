import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

classes_imagenet10 = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church',
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute',
]
wnids = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
         'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class ImageNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # Flat image and label folders, exactly like the week 2 Dataset.
        self.image_dir = Path(root_dir) / f'{split} images'
        self.label_dir = Path(root_dir) / f'{split} labels'
        self.image_paths = sorted(self.image_dir.glob('*.JPEG'))
        self.transform = transform
        if not self.image_paths:
            raise FileNotFoundError(f'No images in {self.image_dir}. Run prepare_data.py first.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f'{image_path.stem}.txt'
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        label = int(label_path.read_text(encoding='utf-8').strip())
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transform(training=False):
    if training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


def get_dataloaders(root_dir, batch_size=32, num_workers=0, seed=42):
    train_dataset = ImageNet10Dataset(root_dir, 'train', get_transform(True))
    valid_dataset = ImageNet10Dataset(root_dir, 'validation', get_transform(False))
    test_dataset = ImageNet10Dataset(root_dir, 'test', get_transform(False))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root = "E:/data/BMI701/ImageNet10"
    train_loader, valid_loader, test_loader = get_dataloaders(
        root, batch_size=20
    )

    # train_loader 设置了 shuffle=True，取出一批随机图片
    images, labels = next(iter(train_loader))

    print('Train / validation / test:',
          len(train_loader.dataset),
          len(valid_loader.dataset),
          len(test_loader.dataset))
    print('Batch images:', images.shape)
    print('Batch labels:', labels.shape)

    # 将归一化后的图片还原，方便正常显示颜色
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))

    for i in range(20):
        image = images[i] * std + mean
        image = image.clamp(0, 1)

        # PyTorch: [C, H, W] → matplotlib: [H, W, C]
        image = image.permute(1, 2, 0).numpy()

        ax = axes.flat[i]
        ax.imshow(image)
        ax.set_title(classes_imagenet10[labels[i].item()])
        ax.axis('off')

    plt.tight_layout()
    plt.show()