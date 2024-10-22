# Set PYTHONPATH to include the src directory
PYTHONPATH := $(shell pwd)/src

.PHONY: check_dataset
check_dataset:
	PYTHONPATH=$(PYTHONPATH) python src/model/cnn/check_dataset.py

.PHONY: train_model
train_model:
	PYTHONPATH=$(PYTHONPATH) python src/model/cnn/train.py

.PHONY: test_model
test_model:
	PYTHONPATH=$(PYTHONPATH) python src/model/cnn/test.py
