# Makefile

.PHONY: setup run-etl run-scoring run-dashboard test clean clear reset install freeze

# Setup the project
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Run ETL pipeline
run-etl:
	python -m src.data.pipeline

# Run scoring pipeline
run-scoring:
	python -m src.models.scoring_pipeline

# Run Streamlit dashboard
run-dashboard:
	streamlit run src/dashboard/dashboard.py

# Run tests
test:
	python -m unittest discover tests

# Remove virtual environment
clear:
	rm -rf venv

# Clean up generated files and virtual environment
clean:
	clear
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Reset the project: clean and setup
reset: clean setup

# Install dependencies
install:
	pip install -r requirements.txt

# Update requirements.txt
freeze:
	pip freeze > requirements.txt