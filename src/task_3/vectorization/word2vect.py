import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from task_3.preprocessing import load_local_lemma_datasets


def create_word2vec_dataset(
    df, text_column, label_column, vector_size=100, window=5, min_count=1
):
    """
    Create Word2Vec vectors from texts and combine them with sentiment labels.
    """
    # Convert texts to list of word lists if they're strings
    texts = df[text_column].tolist()
    if isinstance(texts[0], str):
        texts = [text.split() for text in texts]

    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )

    # Function to create document vector by averaging word vectors
    def get_document_vector(text):
        if isinstance(text, str):
            text = text.split()
        vectors = []
        for word in text:
            if word in w2v_model.wv:
                vectors.append(w2v_model.wv[word])
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(vector_size)

    # Create document vectors
    document_vectors = [get_document_vector(text) for text in df[text_column]]

    # Convert to DataFrame
    vector_columns = [f"feature_{i}" for i in range(vector_size)]
    result_df = pd.DataFrame(document_vectors, columns=vector_columns)
    result_df["sentiment"] = df[label_column]

    return result_df, w2v_model


def get_vector_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset, test_dataset = load_local_lemma_datasets()

    train_dataset, _ = create_word2vec_dataset(
        df=train_dataset, text_column="text", label_column="label"
    )

    test_dataset, _ = create_word2vec_dataset(
        df=test_dataset, text_column="text", label_column="label"
    )

    return train_dataset, test_dataset
