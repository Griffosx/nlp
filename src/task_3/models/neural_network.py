import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from task_3.vectorization.word2vect import get_vector_datasets


class SentimentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SentimentClassifier(nn.Module):
    def __init__(self, input_size=100):
        super(SentimentClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(train_df, batch_size=32, learning_rate=0.001, num_epochs=100):
    # Prepare data
    x_train = train_df.drop("sentiment", axis=1).values
    y_train = train_df["sentiment"].values
    train_dataset = SentimentDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SentimentClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )

    return model


def evaluate_model(model, test_df, batch_size=32):
    # Prepare data
    x_test = test_df.drop("sentiment", axis=1).values
    y_test = test_df["sentiment"].values
    test_dataset = SentimentDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")


def train_and_evaluate():
    train_df, test_df = get_vector_datasets()
    model = train_model(train_df)
    evaluate_model(model, test_df)
