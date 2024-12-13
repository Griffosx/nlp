from model.cnn.dataset import get_test_filenames


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
    main_check_split_consistency()
