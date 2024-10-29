import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model.cnn.dataset import get_data_loaders
from model.cnn.model import load_model


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


def main_test():
    directories = {
        "no_noise": "spectrograms",
        "noise_40db": "spectrograms_with_noise_40",
        "noise_20db": "spectrograms_with_noise_20",
    }
    for noise_name, root_dir in directories.items():
        print(f"Testing with {noise_name.replace('_', ' ')}...")
        _, testloader, classes, idx_to_class = get_data_loaders(root_dir)
        num_classes = len(classes)

        # Load the model
        net = load_model(num_classes, model_path="output_data/model.pth")

        # Test the network and plot confusion matrix
        test_model(net, testloader, idx_to_class, noise_name)


if __name__ == "__main__":
    main_test()
