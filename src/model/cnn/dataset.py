import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.labels = [
            self.label_func(os.path.basename(path)) for path in self.image_paths
        ]
        # Build a label to index mapping
        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted(set(self.labels)))
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.targets = [self.label_to_idx[label] for label in self.labels]

    def label_func(self, fname):
        """Extracts the label from the filename by splitting on the underscore."""
        return fname.split("_")[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(root_dir, batch_size=4, num_workers=2, seed=42, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets with a fixed seed and
    ensures that each class is equally represented in both sets (stratified split).
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ]
    )

    # Initialize the dataset
    dataset = SpectrogramDataset(root_dir=root_dir, transform=transform)

    # Extract targets for stratification
    targets = dataset.targets

    # Initialize StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)

    # Obtain train and test indices
    train_indices, test_indices = next(sss.split(np.zeros(len(targets)), targets))

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoaders
    trainloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Retrieve class mappings
    classes = dataset.label_to_idx
    idx_to_class = dataset.idx_to_label

    return trainloader, testloader, classes, idx_to_class


def get_test_filenames(root_dir, batch_size=4, num_workers=2, seed=42, train_ratio=0.8):
    """
    Retrieves the set of filenames in the training subset.
    """
    dataset = SpectrogramDataset(root_dir=root_dir)
    _, testloader, _, _ = get_data_loaders(
        root_dir, batch_size, num_workers, seed, train_ratio
    )
    return set(
        [dataset.image_paths[i].split("/")[-1] for i in testloader.dataset.indices]
    )
