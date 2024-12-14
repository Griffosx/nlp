from task_3.preprocessing import (
    load_imdb_dataset,
    load_and_clean_imdb_dataset,
    save_raw_datasets_to_local,
    load_local_raw_datasets,
    save_lemma_datasets_to_local,
    load_local_lemma_datasets,
    generate_wordcloud,
)

from task_3.vectorization import get_vector_datasets
from task_3.models.neural_network import train_and_evaluate
from task_3.models.classical_models import classical_models_comparison


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
    # load_imdb_dataset(print_stats=True)
    generate_wordcloud(lemmatisation=False)
    # save_raw_datasets_to_local(num_samples=10000)
    # save_lemma_datasets_to_local()
    # get_vector_datasets()
    # train_and_evaluate()
    # classical_models_comparison()
