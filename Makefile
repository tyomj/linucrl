SHELL := /bin/bash
ENV_PATH := env/bin/activate

.DEFAULT: help

help:
	@echo "     USAGE"
	@echo "==============="
	@echo "make -B download_dataset"
	@echo "       Download Movielens dataset"
	@echo "make -B train"
	@echo "       Starts train.py"

download_dataset:
	@python3 lucrl/utils/download_movielens.py #-c <some_path>

train:
	@python3 lucrl/scripts/train.py

