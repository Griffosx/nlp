import nltk
from nltk.corpus import gutenberg


class GutenbergLoader:
    def __init__(self):
        # Ensure the Gutenberg dataset is downloaded
        nltk.download("gutenberg")

    def list_available_works(self) -> list[str]:
        """List all available works in the Gutenberg corpus."""
        return gutenberg.fileids()

    def list_author_works(self, author: str) -> list[str]:
        """List all works by a specific author."""
        return [
            work
            for work in self.list_available_works()
            if author.lower() in work.lower()
        ]

    def load_chesterton_works(self) -> dict[str, str]:
        """Load Chesterton's major works from the Gutenberg corpus."""
        chesterton_works = {
            "ball": "chesterton-ball.txt",
            "brown": "chesterton-brown.txt",
            "thursday": "chesterton-thursday.txt",
        }

        loaded_works = {}
        for title, filename in chesterton_works.items():
            try:
                text = gutenberg.raw(filename)
                loaded_works[title] = text
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        return loaded_works

    def get_combined_text(self, works: dict[str, str]) -> str:
        """Combine all loaded works into a single text."""
        return "\n\n".join(works.values())

    def get_chesterton_combined_text(self) -> str:
        """Load and combine Chesterton's major works."""
        works = self.load_chesterton_works()
        return self.get_combined_text(works)

    def print_corpus_stats(self, works: dict[str, str]) -> None:
        """Print basic statistics about the loaded works."""
        print("\nCorpus Statistics:")
        total_chars = 0
        total_words = 0
        total_lines = 0

        for title, text in works.items():
            num_chars = len(text)
            num_words = len(text.split())
            num_lines = len(text.splitlines())

            total_chars += num_chars
            total_words += num_words
            total_lines += num_lines

            print(f"\n{title.title()}:")
            print(f"Characters: {num_chars:,}")
            print(f"Words: {num_words:,}")
            print(f"Lines: {num_lines:,}")

        print("\nTotal Statistics:")
        print(f"Total Characters: {total_chars:,}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Lines: {total_lines:,}")

    def show_unique_characters(self, text: str) -> None:
        """Display all unique characters in the text."""
        unique_chars = sorted(list(set(text)))
        print("\nUnique Characters in Corpus:")
        print(f"Number of unique characters: {len(unique_chars)}")
        print("Characters: ", " ".join(unique_chars))


def print_chesterton_info():
    # Initialize the loader
    loader = GutenbergLoader()

    # List all Chesterton works available
    print("Available Chesterton works in Gutenberg:")
    chesterton_works = loader.list_author_works("chesterton")
    for work in chesterton_works:
        print(f"- {work}")

    # Load Chesterton's works
    print("\nLoading Chesterton's works...")
    works = loader.load_chesterton_works()

    # Print statistics for each work
    loader.print_corpus_stats(works)

    # Get combined text
    combined_text = loader.get_combined_text(works)

    # Show unique characters
    loader.show_unique_characters(combined_text)

    # Print a sample
    print("\nSample from combined text:")
    print(combined_text[:500] + "...")
