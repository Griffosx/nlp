from task_3.vectorization.bow import get_vector_datasets as get_vector_datasets_bow
from task_3.vectorization.word2vect import (
    get_vector_datasets as get_vector_datasets_word2vect,
)


def get_vector_datasets(vectorization_type="bow"):
    if vectorization_type == "bow":
        return get_vector_datasets_bow("tfidf")
    else:
        return get_vector_datasets_word2vect()
