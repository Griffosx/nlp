import torch
import torch.nn as nn
import torch.nn.functional as F


def define_model(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
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


def save_model(net, model_path="model.pth"):
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(num_classes, model_path="model.pth"):
    net = define_model(num_classes)
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return net
