from fastai.vision.all import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch


def label_func(fname):
    """Extracts the label from the filename by splitting on the underscore."""
    return fname.name.split("_")[0]


def main():
    # ----------------------------
    # 1. Device Configuration
    # ----------------------------
    # Detect and set the appropriate device (MPS for Mac M1/M2, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set the device in fastai's defaults
    defaults.device = device
    print(f"Using device: {defaults.device}")

    # ----------------------------
    # 2. Data Loading and Preprocessing
    # ----------------------------
    path = Path("spectrograms")  # Ensure the path is correct

    # Verify the dataset directory exists
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory '{path}' does not exist.")

    # Create DataLoaders with appropriate transformations
    dls = ImageDataLoaders.from_path_func(
        path=path,
        fnames=get_image_files(path),
        label_func=label_func,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224, method=ResizeMethod.Squish),  # Ensure exact size
        batch_tfms=[
            *aug_transforms(
                size=224,
                max_warp=0,  # Disable warping to prevent size changes
                max_rotate=10,  # Limit rotation to prevent dimension issues
                max_zoom=1.1,  # Limit zoom to maintain size
                min_scale=1.0,  # Prevent scaling smaller than original
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
        device=device,  # Explicitly set the device
    )

    print(f"Number of training samples: {len(dls.train_ds)}")
    print(f"Number of validation samples: {len(dls.valid_ds)}")
    print(f"Classes: {dls.vocab}")

    # ----------------------------
    # 3. Data Visualization (Optional)
    # ----------------------------
    # Display a batch of training images
    dls.show_batch(max_n=8, figsize=(12, 8))
    plt.show()

    # ----------------------------
    # 4. Model Initialization
    # ----------------------------
    _num_classes = len(dls.vocab)
    learn = cnn_learner(
        dls,
        resnet18,
        metrics=accuracy,
        model_dir="models",  # Directory to save models
        wd=1e-3,  # Weight decay for regularization
    )

    print(learn.model)

    # ----------------------------
    # 5. Learning Rate Finder
    # ----------------------------
    # Uncomment the following lines to find an optimal learning rate
    # learn.lr_find()
    # learn.recorder.plot_lr_find()
    # plt.show()

    # ----------------------------
    # 6. Model Training
    # ----------------------------
    # Fine-tune the model for a specified number of epochs
    num_epochs = 15
    learn.fine_tune(num_epochs)

    # ----------------------------
    # 7. Model Evaluation
    # ----------------------------
    # Generate classification interpretation
    interp = ClassificationInterpretation.from_learner(learn)

    # Plot the confusion matrix
    interp.plot_confusion_matrix(figsize=(10, 8), dpi=100)
    plt.show()

    # Retrieve predictions and true labels
    preds, y_true = learn.get_preds()
    y_pred = preds.argmax(dim=1)

    # Move tensors to CPU and ensure they are float32
    y_true = y_true.cpu().to(torch.float32)
    y_pred = y_pred.cpu().to(torch.float32)

    # Convert to numpy arrays with float32 dtype
    y_true_np = y_true.numpy().astype(np.float32)
    y_pred_np = y_pred.numpy().astype(np.float32)

    # Generate the classification report
    print("Classification Report:")
    print(classification_report(y_true_np, y_pred_np, target_names=dls.vocab))

    # ----------------------------
    # 8. Model Saving
    # ----------------------------
    # Save the trained model's weights
    learn.save("resnet18_spectrograms")
    print("Model saved to 'models/resnet18_spectrograms.pth'")

    # ----------------------------
    # 9. Model Loading (For Inference or Further Training)
    # ----------------------------
    # To load the model later:
    # learn_loaded = cnn_learner(
    #     dls,
    #     resnet18,
    #     metrics=accuracy,
    #     model_dir='models'
    # )
    # learn_loaded.load('resnet18_spectrograms')
    # print("Model loaded from 'models/resnet18_spectrograms.pth'")


if __name__ == "__main__":
    main()
