.PHONY: dev train test deploy clean setup fix-docker docker-diagnose

setup:
	chmod +x scripts/setup_venvs.sh
	./scripts/setup_venvs.sh

fix-docker:
	chmod +x scripts/fix-docker-credentials.sh
	./scripts/fix-docker-credentials.sh

docker-diagnose:
	chmod +x scripts/docker-diagnostic.sh
	./scripts/docker-diagnostic.sh

dev:
	docker-compose -f infrastructure/docker-compose.dev.yml up --build

train:
	source .venv/model/bin/activate && \
	python model/training/train.py

test:
	source .venv/model/bin/activate && \
	pytest tests/

deploy:
	docker-compose -f infrastructure/docker-compose.yml up --build

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +
	rm -rf .venv
