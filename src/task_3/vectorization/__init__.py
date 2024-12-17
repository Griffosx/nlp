from task_3.vectorization.bow import get_vector_datasets as get_vector_datasets_bow
from task_3.vectorization.word2vect import (
    get_vector_datasets as get_vector_datasets_word2vect,
)


def get_vector_datasets(vectorization_type="tfidf"):
    vectorization_function = {
        "count": lambda: get_vector_datasets_bow("count"),
        "tfidf": lambda: get_vector_datasets_bow("tfidf"),
        "word2vect": get_vector_datasets_word2vect,
    }
    return vectorization_function[vectorization_type]()
