import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datasets import load_dataset
from task_3.constants import (
    nlp,
    POSITIVE_LABEL,
    TRAIN_RAW_DATA_PATH,
    TEST_RAW_DATA_PATH,
    TRAIN_LEMMA_DATA_PATH,
    TEST_LEMMA_DATA_PATH,
)


def load_imdb_dataset(
    num_samples=None, print_stats=False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the IMDB movie reviews dataset for sentiment analysis.
    """
    # Load the dataset
    dataset = load_dataset("imdb")

    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Sample if specified
    if num_samples:
        train_df = train_df.sample(min(num_samples, len(train_df)), random_state=42)
        test_df = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    if print_stats:
        # Add some basic statistics
        print(f"Dataset Statistics:")
        print(f"Training samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")
        print(f"\nClass distribution in training:")
        print(train_df["label"].value_counts(normalize=True))
        # Calculate average review length
        train_df["review_length"] = train_df["text"].str.len()
        print(
            f"\nAverage review length: {train_df['review_length'].mean():.0f} characters"
        )

    return train_df, test_df


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the dataset
    cleaned_dataset = dataset.copy()

    # Remove missing values and empty cells
    cleaned_dataset = cleaned_dataset.dropna(subset=["text"])
    cleaned_dataset = cleaned_dataset[cleaned_dataset["text"].str.strip().astype(bool)]

    # Remove HTML tags
    cleaned_dataset["text"] = cleaned_dataset["text"].str.replace(
        r"<[^>]*>", "", regex=True
    )

    # Remove puntuation, leave only alphanumeric characters using regex
    cleaned_dataset["text"] = cleaned_dataset["text"].str.replace(
        r"[^\w\s]", "", regex=True
    )

    # Convert to lowercase
    cleaned_dataset["text"] = cleaned_dataset["text"].str.lower()

    # Remove punctuation and stopwords using spaCy
    def clean_text(text):
        doc = nlp(text)
        # Keep only non-stopword tokens and strip spaces
        return " ".join(token.text.strip() for token in doc if not token.is_stop)

    cleaned_dataset["text"] = cleaned_dataset["text"].apply(clean_text)

    return cleaned_dataset


def load_and_clean_imdb_dataset(
    num_samples=None, print_stats=False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = load_imdb_dataset(num_samples, print_stats)
    return clean_dataset(train_data), clean_dataset(test_data)


def save_raw_datasets_to_local(num_samples=None, print_stats=False):
    train_data, test_data = load_and_clean_imdb_dataset(num_samples, print_stats)
    train_data.to_csv(TRAIN_RAW_DATA_PATH, index=False)
    test_data.to_csv(TEST_RAW_DATA_PATH, index=False)


def load_local_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(TRAIN_RAW_DATA_PATH)
    test_data = pd.read_csv(TEST_RAW_DATA_PATH)
    return train_data, test_data


def save_lemma_datasets_to_local():
    train_data, test_data = load_local_raw_datasets()
    train_data["text"] = train_data["text"].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x)])
    )
    test_data["text"] = test_data["text"].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x)])
    )
    train_data.to_csv(TRAIN_LEMMA_DATA_PATH, index=False)
    test_data.to_csv(TEST_LEMMA_DATA_PATH, index=False)


def load_local_lemma_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(TRAIN_LEMMA_DATA_PATH)
    test_data = pd.read_csv(TEST_LEMMA_DATA_PATH)
    return train_data, test_data


def generate_wordcloud(lemmatisation=True):
    """
    For each label in the dataset, generate and plot a wordcloud.
    """
    if lemmatisation:
        dataset, _ = load_local_lemma_datasets()
    else:
        dataset, _ = load_local_raw_datasets()

    # Get unique labels from the dataset
    unique_labels = dataset["label"].unique()

    # Create a figure with subplots for each label
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(15, 5))

    for idx, label in enumerate(unique_labels):
        # Filter text for current label
        texts = dataset[dataset["label"] == label]["text"]

        if lemmatisation:
            # Apply lemmatization using spaCy
            processed_texts = []
            for text in texts:
                doc = nlp(text)
                lemmatized = " ".join([token.lemma_ for token in doc])
                processed_texts.append(lemmatized)
            text = " ".join(processed_texts)
        else:
            text = " ".join(texts)

        # Generate wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )

        # Plot wordcloud
        if len(unique_labels) > 1:
            axes[idx].imshow(wordcloud)
            axes[idx].axis("off")
            axes[idx].set_title(
                f"Label: {'positive' if label == POSITIVE_LABEL else 'negative'}"
            )
        else:
            axes.imshow(wordcloud)
            axes.axis("off")
            axes.set_title(
                f"Label: {'positive' if label == POSITIVE_LABEL else 'negative'}"
            )

    plt.tight_layout()
    plt.show()
