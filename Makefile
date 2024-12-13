# Set PYTHONPATH to include the src directory
PYTHONPATH := $(shell pwd)/src

.PHONY: check_dataset
check_dataset:
	PYTHONPATH=$(PYTHONPATH) python src/task_1_and_1/model/cnn/check_dataset.py

.PHONY: train_model
train_model:
	PYTHONPATH=$(PYTHONPATH) python src/task_1_and_1/model/cnn/train.py

.PHONY: test_model
test_model:
	PYTHONPATH=$(PYTHONPATH) python src/task_1_and_1/model/cnn/test.py

.PHONY: task_3
task_3:
	PYTHONPATH=$(PYTHONPATH) python src/task_3/main.py
