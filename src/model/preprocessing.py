import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def remove_silence(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove silence from the beginning and end of the audio data.
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
    """
    hamming_window = np.hamming(len(frame))
    windowed_frame = frame * hamming_window
    return windowed_frame


def divide_in_frames(audio_path: str) -> np.ndarray:
    """
    Divide the audio file into frames after removing silence from the beginning and end.
    """
    # Read the audio data and sampling rate
    _sampling_rate, audio_data = wavfile.read(audio_path)

    # Old method to calculate the number of samples per frame
    # samples_per_frame = int(sampling_rate * frame_duration_ms / 1000)
    # So the old samples per frame was (44100 * 30) / 1000 = 1323
    samples_per_frame = 1024
    # 1024 samples per frame means that the frame duration is circa 24 ms

    # Remove silence
    trimmed_audio = remove_silence(audio_data)

    # Reshape the audio data into frames
    num_frames = len(trimmed_audio) // samples_per_frame
    frames = np.reshape(
        trimmed_audio[: num_frames * samples_per_frame], (num_frames, samples_per_frame)
    )

    return [apply_hamming_window(frame) for frame in frames]


def add_noise(audio_data: np.ndarray, snr_db: int = 20) -> np.ndarray:
    """
    Add white Gaussian noise to the audio data at a specified SNR level.
    """
    # Convert audio data to float
    audio_data_float = audio_data.astype(np.float32)

    # Remove silence to compute signal power
    trimmed_audio = remove_silence(audio_data_float)

    # Calculate the power of the non-silent part of the original signal
    signal_power = np.mean(trimmed_audio**2)

    # Compute noise power based on desired SNR
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(trimmed_audio))

    # Add noise to the original signal
    noisy_audio = trimmed_audio + noise

    return noisy_audio


def _magnitute_spectrum(frame: np.ndarray) -> np.ndarray:
    """
    Calculate magnitude spectrum from a given audio frame using the Short-Time Fourier Transform (STFT).
    """
    fft_result = np.fft.fft(frame)
    magnitude_spectrum = np.abs(
        fft_result[: len(fft_result) // 2]
    )  # Keep only the positive frequencies
    return magnitude_spectrum


def generate_spectrums(audio_path: str) -> np.ndarray:
    """
    Generate a spectrogram from an audio file after removing silence.
    """
    # Divide the audio into frames after removing silence and apply Hamming window
    frames = divide_in_frames(audio_path)

    # Create spectrums for each frame
    spectrums = [_magnitute_spectrum(frame) for frame in frames]

    # Stack the spectrograms into a 2D array (frequency x time)
    spectrums_2d = np.array(spectrums).T

    return spectrums_2d


def save_spectrogram_image(spectrums_2d: np.ndarray, filename: str, axes: bool) -> None:
    """
    Save the 2D spectrogram image with optional axes, labels, titles, and colorbar.
    """
    # Create a figure with specified size
    plt.figure(figsize=(10, 10))

    # Display the spectrogram
    plt.imshow(
        10 * np.log10(spectrums_2d + 1e-10),
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


def generate_and_save_noisy_audio(audio_dir: str, snr_db_list: list[int]):
    """
    Generate noisy audio files for each WAV file in the audio directory
    at specified SNR levels and save them to relative folders.
    """
    for snr_db in snr_db_list:
        # Create the output directory
        output_dir = f"{audio_dir}_with_noise_{snr_db}"
        os.makedirs(output_dir, exist_ok=True)

        # Find all WAV files in the audio directory
        wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))

        for audio_path in wav_files[:10]:
            # Read the audio data and sampling rate
            sampling_rate, audio_data = wavfile.read(audio_path)
            trimmed_audio_data = remove_silence(audio_data)

            # Add noise to audio data
            noisy_audio_data = add_noise(trimmed_audio_data, snr_db)

            # Ensure data is in appropriate format
            if audio_data.dtype == np.int16:
                # Clip the data to int16 range
                noisy_audio_data = np.clip(noisy_audio_data, -32768, 32767)
                # Convert back to int16
                noisy_audio_data = noisy_audio_data.astype(np.int16)
            elif audio_data.dtype == np.int32:
                # Clip and convert to int32
                noisy_audio_data = np.clip(noisy_audio_data, -2147483648, 2147483647)
                noisy_audio_data = noisy_audio_data.astype(np.int32)
            else:
                # For other types, perhaps save as float32
                noisy_audio_data = noisy_audio_data.astype(np.float32)

            # Define the output file path
            base_name = os.path.basename(audio_path)
            output_path = os.path.join(output_dir, base_name)

            # Save the noisy audio data
            wavfile.write(output_path, sampling_rate, noisy_audio_data)
            print(f"Saved noisy audio file {output_path} with SNR {snr_db} dB")


def generate_and_save_all_spectrograms(audio_dir: str, snr_db_list: list[int] = None):
    """
    Generate spectrograms for all WAV files in the given audio directory
    and save them in the 'spectrograms' directory, including versions with added noise.
    """
    if snr_db_list is None:
        snr_db_list = [0]  # 0 indicates original audio without added noise

    # frame_duration_ms = 24

    for snr_db in snr_db_list:
        if snr_db == 0:
            input_dir = audio_dir
            spectrogram_dir = "spectrograms"
        else:
            input_dir = f"{audio_dir}_with_noise_{snr_db}"
            spectrogram_dir = f"spectrograms_with_noise_{snr_db}"

        # Create the spectrogram directory if it doesn't exist
        os.makedirs(spectrogram_dir, exist_ok=True)

        # Find all WAV files in the input directory
        wav_files = glob.glob(os.path.join(input_dir, "*.wav"))

        for audio_path in wav_files:
            # Generate the spectrogram
            spectrogram = generate_spectrums(audio_path)

            # Extract the base filename without extension
            base_name = os.path.splitext(os.path.basename(audio_path))[0]

            # Define the path for the spectrogram image
            spectrogram_path = os.path.join(spectrogram_dir, f"{base_name}.png")

            # Save the spectrogram image
            save_spectrogram_image(spectrogram, spectrogram_path, axes=False)
            print(f"Saved spectrogram image {spectrogram_path}")


if __name__ == "__main__":
    # Generate noisy audio files with SNR levels of 20 dB and 40 dB
    generate_and_save_noisy_audio("audio", [20, 40])

    # Generate spectrograms for the original and noisy audio files
    generate_and_save_all_spectrograms("audio", snr_db_list=[0, 20, 40])
