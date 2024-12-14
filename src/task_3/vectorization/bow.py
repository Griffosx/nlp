from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from task_3.preprocessing import load_local_lemma_datasets


def create_bow_datasets(
    train_dataset,
    test_dataset,
    vectorizer_type="count",
    max_features=1000,
    ngram_range=(1, 1),
):
    """
    Create BOW vectors using either CountVectorizer or TfidfVectorizer

    Parameters:
    - vectorizer_type: 'count' for regular BOW, 'tfidf' for TF-IDF weighted
    - max_features: maximum number of features to keep
    - ngram_range: tuple (min_n, max_n) representing the range of ngrams
    """
    # Choose vectorizer
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2
        )  # Ignore terms that appear in less than 2 documents
    else:
        vectorizer = CountVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2
        )

    # Fit and transform training data
    X_train = vectorizer.fit_transform(train_dataset["text"])

    # Transform test data
    X_test = vectorizer.transform(test_dataset["text"])

    # Convert to DataFrames
    feature_names = vectorizer.get_feature_names_out()

    train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    train_df["sentiment"] = train_dataset["label"]

    test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    test_df["sentiment"] = test_dataset["label"]

    # Print some information about the vectorization
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix shape: {X_train.shape}")
    print("\nMost common terms:")
    if vectorizer_type == "count":
        term_frequencies = X_train.sum(axis=0).A1
        top_terms = sorted(
            zip(vectorizer.get_feature_names_out(), term_frequencies),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for term, freq in top_terms:
            print(f"{term}: {freq}")

    return train_df, test_df


def get_vector_datasets(vectorization_type="count"):
    train_dataset, test_dataset = load_local_lemma_datasets()

    train_dataset, test_dataset = create_bow_datasets(
        train_dataset, test_dataset, vectorization_type
    )

    return train_dataset, test_dataset
