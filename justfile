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
