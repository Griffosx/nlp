import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def label_func(fname):
    """Extracts the label from the filename by splitting on the underscore."""
    return fname.split("_")[0]


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.labels = [label_func(os.path.basename(path)) for path in self.image_paths]
        # Build a label to index mapping
        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted(set(self.labels)))
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.targets = [self.label_to_idx[label] for label in self.labels]

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


def define_model(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 72 * 72, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 72 * 72)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    return net


def train_model(net, trainloader, criterion, optimizer, num_epochs=2):
    net.train()  # Set the model to training mode
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Optionally, print statistics
            if i % 50 == 49:  # Print every 50 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0
    print("Finished Training")


def save_model(net, model_path="model.pth"):
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(num_classes, model_path="model.pth"):
    net = define_model(num_classes)
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return net


def test_model(net, testloader, idx_to_class, noise_name):
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images ({noise_name}): {accuracy:.2f}%")
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, idx_to_class, noise_name)
    return accuracy


def plot_confusion_matrix(labels, preds, idx_to_class, noise_name):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(idx_to_class.values())
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot as an image
    save_path = f"output_data/confusion_matrix_{noise_name}"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # Close the plot to prevent display
    print(f"Confusion matrix saved to {save_path}")


def main_train():
    root_dir = "spectrograms"
    trainloader, _testloader, classes, _idx_to_class = get_data_loaders(root_dir)
    num_classes = len(classes)
    net = define_model(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    train_model(net, trainloader, criterion, optimizer, num_epochs=15)

    # Save the model
    save_model(net, model_path="output_data/model.pth")


def main_test():
    directories = {
        "no_noise": "spectrograms",
        "noise_20db": "spectrograms_with_noise_20",
        "noise_40db": "spectrograms_with_noise_40",
    }
    for noise_name, root_dir in directories.items():
        print(f"Testing with {noise_name.replace('_', ' ')}...")
        _, testloader, classes, idx_to_class = get_data_loaders(root_dir)
        num_classes = len(classes)

        # Load the model
        net = load_model(num_classes, model_path="output_data/model.pth")

        # Test the network and plot confusion matrix
        test_model(net, testloader, idx_to_class, noise_name)


def main_check_split_consistency():
    directories = {
        "original": "spectrograms",
        "20dB_noise": "spectrograms_with_noise_20",
        "40dB_noise": "spectrograms_with_noise_40",
    }

    # Extract training filenames for each directory
    train_filenames = {}
    for key, dir_path in directories.items():
        train_filenames[key] = get_test_filenames(dir_path)
        print(f"Number of training samples in {key}: {len(train_filenames[key])}")

    # Compare filenames between directories
    original_train = train_filenames["original"]
    noise_20_train = train_filenames["20dB_noise"]
    noise_40_train = train_filenames["40dB_noise"]

    # Check if all training sets have the same filenames
    consistent_20 = original_train == noise_20_train
    consistent_40 = original_train == noise_40_train

    print(f"Train split consistent between original and 20 dB noise: {consistent_20}")
    print(f"Train split consistent between original and 40 dB noise: {consistent_40}")

    # Optionally, find differences
    if not consistent_20:
        diff_20 = original_train.symmetric_difference(noise_20_train)
        print(f"Differences between original and 20 dB noise train sets: {diff_20}")

    if not consistent_40:
        diff_40 = original_train.symmetric_difference(noise_40_train)
        print(f"Differences between original and 40 dB noise train sets: {diff_40}")


if __name__ == "__main__":
    main_train()
    main_test()
    main_check_split_consistency()
