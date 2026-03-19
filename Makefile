.PHONY: install test lint clean

install:
	pip install -r requirements.txt

test:
	python -m pytest 03_code/operators/tests/ -v

lint:
	python -m flake8 03_code/ 04_experiments/ --max-line-length=120

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
