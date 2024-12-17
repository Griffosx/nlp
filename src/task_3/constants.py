import spacy

SPACY_MODEL = "en_core_web_sm"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1

# Download spaCy model
spacy.cli.download(SPACY_MODEL)
# Load spaCy model only once
nlp = spacy.load(SPACY_MODEL)

DATASETS_BASE_PATH = "assets/task_3/datasets"
TRAIN_RAW_DATA_PATH = f"{DATASETS_BASE_PATH}/raw/train.csv"
TEST_RAW_DATA_PATH = f"{DATASETS_BASE_PATH}/raw/test.csv"
TRAIN_LEMMA_DATA_PATH = f"{DATASETS_BASE_PATH}/lemma/train.csv"
TEST_LEMMA_DATA_PATH = f"{DATASETS_BASE_PATH}/lemma/test.csv"
