import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.backends.mps
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils

from sklearn.metrics import classification_report, confusion_matrix


# Define the custom Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, image_files, classes, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.image_files = image_files
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB
        label_name = img_name.split("_")[0]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(dataset_dir, train_transforms, val_transforms, batch_size=32):
    # Get all image files
    image_files = [
        f
        for f in os.listdir(dataset_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    # Define classes and class_to_idx
    classes = sorted(list(set([f.split("_")[0] for f in image_files])))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Shuffle image files
    random.shuffle(image_files)

    # Split into train and val
    train_size = int(0.8 * len(image_files))
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # Create datasets
    train_dataset = SpectrogramDataset(
        root_dir=dataset_dir,
        image_files=train_files,
        classes=classes,
        class_to_idx=class_to_idx,
        transform=train_transforms,
    )
    val_dataset = SpectrogramDataset(
        root_dir=dataset_dir,
        image_files=val_files,
        classes=classes,
        class_to_idx=class_to_idx,
        transform=val_transforms,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    return dataloaders, dataset_sizes, classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
    std = np.array([0.229, 0.224, 0.225])  # ImageNet std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated


def visualize_images(dataloaders, classes):
    inputs, classes_idx = next(iter(dataloaders["train"]))
    out = utils.make_grid(inputs[:8])
    imshow(out, title=[classes[i] for i in classes_idx[:8]])


def initialize_model(num_classes):
    model = models.resnet18(pretrained=True)
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    dataset_sizes,
    device,
    num_epochs=25,
):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it has better accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f"Best Val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Normalize the confusion matrix.
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            f"{cm[i, j]} ({cm_norm[i, j]:.2f})",
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(filename, num_classes, device):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {filename}")
    return model


def train_and_save():
    # Define data directory
    dataset_dir = "spectrograms"  # Corrected spelling

    # Define image transformations
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # ImageNet mean and std
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load data
    dataloaders, dataset_sizes, classes = load_data(
        dataset_dir, train_transforms, val_transforms, batch_size=32
    )

    print(f"Number of training samples: {dataset_sizes['train']}")
    print(f"Number of validation samples: {dataset_sizes['val']}")
    print(f"Classes: {classes}")

    # Visualize some images
    # visualize_images(dataloaders, classes)

    # Initialize model
    num_classes = len(classes)
    model = initialize_model(num_classes)

    # Move model to device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = model.to(device)

    print(model)

    # Define loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 25
    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        dataset_sizes,
        device,
        num_epochs=num_epochs,
    )

    # Evaluate the model
    evaluate_model(model, dataloaders["val"], classes, device)

    # Save the model
    save_model(model, "resnet18_spectrograms.pth")

    # Load the model
    model_loaded = load_model("resnet18_spectrograms.pth", num_classes, device)


if __name__ == "__main__":
    train_and_save()
