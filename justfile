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

# Check code formatting without modifying files
format-check:
	ruff format --check faceberg/

# Run type checking with mypy
type-check:
	mypy faceberg/

# Run all unit tests
test:
	pytest faceberg/tests/ -v

# Run tests with coverage
test-cov:
	pytest faceberg/tests/ --cov=faceberg --cov-report=xml --cov-report=term-missing -v

# Run all checks (format, lint, type-check, and tests)
check: format-check lint type-check test

# Run linting and tests together
check-fast: lint test

# Fix linting issues and run tests
fix: lint-fix test
