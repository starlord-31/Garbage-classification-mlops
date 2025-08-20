.PHONY: install test lint docker

install:
	pip install -r requirements.txt

lint:
	flake8 .

test:
	pytest tests/

docker-build:
	docker build -t garbage-classifier:latest .

docker-run:
	docker run -p 8000:8000 garbage-classifier:latest

clean:
	find . -type f -name '*.pyc' -delete
