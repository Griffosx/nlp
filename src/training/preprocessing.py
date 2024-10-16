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


def generate_spectrogram(audio_path: str, frame_duration_ms: int = 40) -> np.ndarray:
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


def save_spectrogram_image(spectrogram_2d: np.ndarray, filename: str) -> None:
    """
    Save the 2D spectrogram image.

    Parameters:
    spectrogram_2d (np.ndarray): The 2D spectrogram array (frequency x time).
    filename (str): The filename to save the image as.
    """
    # Plot and save the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(
        10 * np.log10(spectrogram_2d + 1e-10),
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.title("Spectrogram")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Frequency Bin")

    # Save the figure as an image file
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory


def generate_all_spectograms():
    audio_files = [
        "audio/analyse_gary_inskeep_12.wav",
        "audio/analyse_christopher_navarrez_10.wav",
        "audio/analyse_darrell_robinson_12.wav",
        "audio/audio_kim_howard_10.wav",
        "audio/audio_kimmy_west_12.wav",
        "audio/audio_leroy_alshak_10.wav",
    ]
    for audio_path in audio_files:
        # Generate the spectrogram
        spectrogram = generate_spectrogram(audio_path)

        # Save the spectrogram image
        image_name = audio_path.split("/")[1].split(".")[0]
        save_spectrogram_image(spectrogram, f"spectograms/{image_name}.png")


if __name__ == "__main__":
    generate_all_spectograms()
