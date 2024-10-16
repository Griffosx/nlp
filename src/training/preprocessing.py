import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def remove_silence(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove silence from the beginning and end of the audio data.

    Parameters:
    audio_data (np.ndarray): The input audio data.
    threshold (float): The amplitude threshold for silence detection.

    Returns:
    np.ndarray: The trimmed audio data.
    """
    abs_audio = np.abs(audio_data)
    mask = abs_audio > threshold * np.max(abs_audio)

    if not np.any(mask):
        return audio_data  # Return original if no non-silent parts found

    first = np.argmax(mask)
    last = len(mask) - np.argmax(mask[::-1])

    return audio_data[first:last]


def apply_hamming_window(frame: np.ndarray) -> np.ndarray:
    """
    Apply the Hamming window to the given frame.

    Parameters:
    frame (np.ndarray): The input audio frame.

    Returns:
    np.ndarray: The windowed frame.
    """
    hamming_window = np.hamming(len(frame))
    windowed_frame = frame * hamming_window
    return windowed_frame


def divide_in_frames(audio_path: str, frame_duration_ms: int = 40) -> np.ndarray:
    """
    Divide the audio file into frames after removing silence from the beginning and end.

    Parameters:
    audio_path (str): Path to the audio file.
    frame_duration_ms (int): Duration of each frame in milliseconds.

    Returns:
    np.ndarray: Array of frames (2D array).
    """
    # Read the audio data and sampling rate
    sampling_rate, audio_data = wavfile.read(audio_path)

    # Remove silence
    trimmed_audio = remove_silence(audio_data)

    # Calculate the number of samples per frame
    samples_per_frame = int(sampling_rate * frame_duration_ms / 1000)

    # Reshape the audio data into frames
    num_frames = len(trimmed_audio) // samples_per_frame
    frames = np.reshape(
        trimmed_audio[: num_frames * samples_per_frame], (num_frames, samples_per_frame)
    )

    return [apply_hamming_window(frame) for frame in frames]


def create_spectrogram(frame: np.ndarray) -> np.ndarray:
    """
    Create a spectrogram from a given audio frame using the Short-Time Fourier Transform (STFT).

    Parameters:
    frame (np.ndarray): The input audio frame.

    Returns:
    np.ndarray: The magnitude spectrogram of the input frame.
    """
    fft_result = np.fft.fft(frame)
    magnitude_spectrum = np.abs(
        fft_result[: len(fft_result) // 2]
    )  # Keep only the positive frequencies
    return magnitude_spectrum


def generate_spectrogram(audio_path: str, frame_duration_ms) -> np.ndarray:
    """
    Generate a spectrogram from an audio file after removing silence.

    Parameters:
    audio_path (str): Path to the audio file.
    frame_duration_ms (int): Duration of each frame in milliseconds.

    Returns:
    np.ndarray: The spectrogram as a 2D array (frequency x time).
    """
    # Divide the audio into frames after removing silence
    frames = divide_in_frames(audio_path, frame_duration_ms)

    # Apply Hamming window and create spectrogram for each frame
    spectrograms = [create_spectrogram(frame) for frame in frames]

    # Stack the spectrograms into a 2D array (frequency x time)
    spectrogram_2d = np.array(spectrograms).T

    return spectrogram_2d


def save_spectrogram_image(
    spectrogram_2d: np.ndarray, filename: str, axes: bool
) -> None:
    """
    Save the 2D spectrogram image with optional axes, labels, titles, and colorbar.

    Parameters:
    spectrogram_2d (np.ndarray): The 2D spectrogram array (frequency x time).
    filename (str): The filename to save the image as.
    axes (bool): Whether to include axes, labels, titles, and colorbar. Defaults to True.
    """
    # Create a figure with specified size
    plt.figure(figsize=(10, 10))

    # Display the spectrogram
    plt.imshow(
        10 * np.log10(spectrogram_2d + 1e-10),
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )

    if axes:
        # Add colorbar with label
        plt.colorbar(label="Magnitude (dB)")

        # Add title and axis labels
        plt.title("Spectrogram")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Frequency Bin")
    else:
        # Remove all axes, ticks, and spines
        plt.axis("off")

    # Save the figure
    if axes:
        # Save with default bounding box and padding
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    else:
        # Save with tight bounding box and no padding
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)

    # Close the figure to free up memory
    plt.close()

    print(
        f"Saved spectrogram image as {filename} with axes={'enabled' if axes else 'disabled'}."
    )


def generate_and_save_all_spectograms():
    """
    Generate spectrograms for all WAV files in the 'audio' directory
    and save them in the 'spectograms' directory.
    """
    audio_dir = "audio"
    spectrogram_dir = "spectrograms"
    frame_duration_ms = 30

    # Create the spectrogram directory if it doesn't exist
    os.makedirs(spectrogram_dir, exist_ok=True)

    # Find all WAV files in the audio directory
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    for audio_path in wav_files:
        # Generate the spectrogram
        spectrogram = generate_spectrogram(audio_path, frame_duration_ms)

        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Define the path for the spectrogram image
        spectrogram_path = os.path.join(spectrogram_dir, f"{base_name}.png")

        # Save the spectrogram image
        save_spectrogram_image(spectrogram, spectrogram_path, axes=False)


if __name__ == "__main__":
    generate_and_save_all_spectograms()
