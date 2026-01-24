# Default recipe to display help
default:
	@just --list

# Run ruff linter on the codebase
lint:
	ruff check faceberg/

# Run ruff linter and automatically fix issues
lint-fix:
	ruff check --fix faceberg/

# Format code with ruff
format:
	ruff format faceberg/

# Run all unit tests
test:
	pytest faceberg/tests/ -v

# Run tests with coverage
test-cov:
	pytest faceberg/tests/ --cov=faceberg --cov-report=term-missing -v

# Run linting and tests together
check: lint test

# Fix linting issues and run tests
fix: lint-fix test
