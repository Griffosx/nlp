import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import gc
from task_4.loader import GutenbergLoader
from task_4.encoder import CharacterEncoder


class TextDataset(Dataset):
    def __init__(self, text: str, char_encoder: CharacterEncoder, seq_length: int):
        self.text = text
        self.char_encoder = char_encoder
        self.seq_length = seq_length
        self.text_indices = [char_encoder.char_to_index[char] for char in text]
        self.num_sequences = len(self.text_indices) - self.seq_length - 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the sequence and target
        sequence = self.text_indices[idx : idx + self.seq_length]
        target = self.text_indices[idx + self.seq_length]

        # Create one-hot encoding for input sequence
        X = torch.zeros(
            self.seq_length, self.char_encoder.vocab_size, dtype=torch.float32
        )
        for t, char_idx in enumerate(sequence):
            X[t, char_idx] = 1

        # Return target as a single integer (not one-hot encoded)
        return X, torch.tensor(target, dtype=torch.long)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        # Return logits (no softmax)
        return self.fc(x[:, -1, :])


class TextGenerator:
    def __init__(
        self,
        seq_length: int = 40,
        device: str = "mps",
    ):
        self.seq_length = seq_length
        self.device = device
        self.model = None

    def build_model(self, vocab_size: int) -> None:
        """Construct the LSTM model."""
        self.model = LSTM(vocab_size).to(self.device)

    def train(
        self,
        text: str,
        char_encoder: CharacterEncoder,
        model_save_path: str = "model.pth",
        encoder_save_path: str = "char_encoder.json",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ) -> List[float]:
        """Train the model and save it to disk."""
        if self.model is None:
            self.build_model(char_encoder.vocab_size)

        dataset = TextDataset(text, char_encoder, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # CrossEntropyLoss expects raw logits and target indices
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                # Get logits from model
                logits = self.model(batch_X)
                # CrossEntropyLoss expects logits and target class indices
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                del batch_X, batch_y, logits, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_count % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_count}/{len(dataloader)}")

            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save model and encoder
        torch.save(self.model.state_dict(), model_save_path)
        char_encoder.save_mappings(encoder_save_path)
        print(f"Model saved to {model_save_path}")
        print(f"Character encoder saved to {encoder_save_path}")

        return losses

    def generate_text(
        self,
        seed_text: str,
        model_load_path: str = "model.pth",
        encoder_load_path: str = "char_encoder.json",
        length: int = 200,
        diversity: float = 0.5,
    ) -> str:
        """Load model and generate text."""
        char_encoder = CharacterEncoder()
        char_encoder.load_mappings(encoder_load_path)

        if self.model is None:
            self.build_model(char_encoder.vocab_size)
            self.model.load_state_dict(
                torch.load(model_load_path, map_location=self.device)
            )

        if len(seed_text) != self.seq_length:
            raise ValueError(f"Seed text must be {self.seq_length} characters long")

        self.model.eval()
        current_sequence = seed_text
        generated_text = seed_text

        with torch.no_grad():
            for _ in range(length):
                # Prepare input tensor
                x_pred = torch.zeros(
                    (1, self.seq_length, char_encoder.vocab_size), dtype=torch.float32
                )
                for t, char in enumerate(current_sequence):
                    if char in char_encoder.char_to_index:
                        x_pred[0, t, char_encoder.char_to_index[char]] = 1

                # Generate prediction
                x_pred = x_pred.to(self.device)
                # Get logits from model
                logits = self.model(x_pred)

                # Apply softmax and temperature scaling for generation
                preds = torch.softmax(logits / diversity, dim=-1).cpu().numpy()[0]

                # Sample next character
                next_index = np.random.choice(len(preds), p=preds)
                next_char = char_encoder.index_to_char[next_index]

                # Update sequences
                generated_text += next_char
                current_sequence = current_sequence[1:] + next_char

                del x_pred, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return generated_text


def train():
    loader = GutenbergLoader()
    encoder = CharacterEncoder()
    generator = TextGenerator(seq_length=40)

    works = loader.load_chesterton_works()
    combined_text = loader.get_combined_text(works)
    encoder.fit(combined_text)

    epochs = 10
    batct_size = 512
    losses = generator.train(
        text=combined_text, char_encoder=encoder, epochs=epochs, batch_size=batct_size
    )
    print("Training completed!")


def generate():
    loader = GutenbergLoader()
    works = loader.load_chesterton_works()
    combined_text = loader.get_combined_text(works)

    generator = TextGenerator(seq_length=40)

    combined_text = "The true soldier fights not because he hates what is in front of him, but because he loves what is behind him."
    seed_text = combined_text[:40].replace("\n", " ").strip()
    print(f"Seed text: {seed_text}")

    print("\nSeed text:")
    print(seed_text)
    print("\nGenerated texts with different diversity levels:")

    for diversity in [0.2, 0.5, 1.0]:
        print(f"\nDiversity: {diversity}")
        print("-" * 50)
        generated = generator.generate_text(
            seed_text=seed_text, length=200, diversity=diversity
        )
        print(generated)
        print("-" * 50)


if __name__ == "__main__":
    # Uncomment the function you want to run:
    # train()  # For training phase
    generate()  # For generation phase
