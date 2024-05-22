from torchvision.datasets import SVHN


class CustomSVHN(SVHN):
    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super().__init__(root, split, transform, target_transform, download)
        self.targets = self.labels

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target
