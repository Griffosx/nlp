import json
from loader import GutenbergLoader


class CharacterEncoder:
    def __init__(self):
        self.char_to_index: dict[str, int] = {}
        self.index_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> None:
        """Create character to index mappings from text."""
        # Get unique characters and sort them for consistency
        unique_chars = sorted(list(set(text)))
        self.vocab_size = len(unique_chars)

        # Create bidirectional mappings
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

    def save_mappings(self, filename: str) -> None:
        """Save character mappings to a JSON file."""
        mappings = {
            "char_to_index": self.char_to_index,
            "index_to_char": {
                str(k): v for k, v in self.index_to_char.items()
            },  # Convert int keys to str for JSON
            "vocab_size": self.vocab_size,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)

    def load_mappings(self, filename: str) -> None:
        """Load character mappings from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        self.char_to_index = mappings["char_to_index"]
        self.index_to_char = {
            int(k): v for k, v in mappings["index_to_char"].items()
        }  # Convert str keys back to int
        self.vocab_size = mappings["vocab_size"]

    def print_mappings(self) -> None:
        """Print character mappings in a readable format."""
        print("\nCharacter to Index Mappings:")
        print("-" * 30)
        for char, idx in sorted(self.char_to_index.items(), key=lambda x: x[1]):
            if char.isspace():
                char_display = f"[space-{ord(char)}]"
            elif char == "\n":
                char_display = "[newline]"
            elif char == "\t":
                char_display = "[tab]"
            else:
                char_display = char
            print(f"'{char_display}' -> {idx}")
        print(f"\nTotal vocabulary size: {self.vocab_size}")


def save_mappings():
    print("Loading Chesterton's works...")
    loader = GutenbergLoader()
    works = loader.load_chesterton_works()
    combined_text = loader.get_combined_text(works)

    # Create and fit the character encoder
    print("\nCreating character encodings...")
    encoder = CharacterEncoder()
    encoder.fit(combined_text)

    # Print the mappings
    encoder.print_mappings()

    # Save the mappings
    encoder.save_mappings("character_mappings.json")
    print("\nMappings saved to 'character_mappings.json'")

    # Example usage
    print("\nExample encodings:")
    sample_text = "Chesterton"
    print(f"\nOriginal text: {sample_text}")
    print("Character indices:", [encoder.char_to_index[c] for c in sample_text])

    # Demonstrate reconstruction
    indices = [encoder.char_to_index[c] for c in sample_text]
    reconstructed = "".join(encoder.index_to_char[i] for i in indices)
    print(f"Reconstructed text: {reconstructed}")
