# Default recipe to display help
default:
	@just --list

# Format code with ruff
format:
	ruff format faceberg/

# Check code formatting without modifying files
check:
	ruff check faceberg/
	ruff format --check faceberg/

# Run all unit tests
test:
	pytest faceberg/tests/ -v

# Run tests with coverage
cov:
	pytest faceberg/tests/ --cov=faceberg --cov-report=xml --cov-report=term-missing -v

# Build distribution packages
build:
	python -m build

# Upload to TestPyPI
publish-test: build
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI
publish: build
	python -m twine upload dist/*
