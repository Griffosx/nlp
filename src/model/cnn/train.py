import torch.nn as nn
import torch.optim as optim
from model.cnn.dataset import get_data_loaders
from model.cnn.model import define_model, save_model


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
            if i % 50 == 49:  # Print every 50 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0
    print("Finished Training")


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


if __name__ == "__main__":
    main_train()
