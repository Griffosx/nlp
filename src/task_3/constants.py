import spacy

SPACY_MODEL = "en_core_web_sm"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1

# Download spaCy model
spacy.cli.download(SPACY_MODEL)
# Load spaCy model only once
nlp = spacy.load(SPACY_MODEL)

DATASETS_PATH = "assets/task_3/datasets"
TRAIN_DATA_PATH = f"{DATASETS_PATH}/train.csv"
TEST_DATA_PATH = f"{DATASETS_PATH}/test.csv"
