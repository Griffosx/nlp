from task_3.preprocessing import (
    load_and_clean_imdb_dataset,
    save_raw_datasets_to_local,
    load_local_raw_datasets,
    generate_wordcloud,
)


def check_load_and_clean():
    # Load a small sample to test
    train_data, test_data = load_and_clean_imdb_dataset(num_samples=10)

    # Display first few examples
    print("Sample reviews before cleaning:")
    for i in range(5):
        print(f"\nReview {i+1}:")
        print(f"Text: {train_data['text'].iloc[i][:200]}...")
        print(
            f"Sentiment: {'Positive' if train_data['label'].iloc[i] == 1 else 'Negative'}"
        )


if __name__ == "__main__":
    generate_wordcloud(lemmatisation=True)
    # save_raw_datasets_to_local(num_samples=100)
