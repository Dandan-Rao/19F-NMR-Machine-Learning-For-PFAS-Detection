# Define the tools 
FORMATTER = nbqa black
LINTER = nbqa pylint
STYLE_CHECKER = nbqa flake8

# Find all Jupyter Notebooks recursively
NOTEBOOKS := $(shell find ./notebooks/ -name "*.ipynb")
PYTHON_SCRIPTS := $(shell find ./src/ -name "*.py")

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


# Format all notebooks
format:
	@echo "Formatting all notebooks..."
	@$(FORMATTER) $(NOTEBOOKS)

	@echo "Formatting all python scripts..."
	@black $(PYTHON_SCRIPTS)


# Lint all notebooks
lint:
	@echo "Linting all notebooks..."
	@$(LINTER) $(NOTEBOOKS)

# Check style for all notebooks
check-style:
	@echo "Checking style for all notebooks..."
	@$(STYLE_CHECKER) $(NOTEBOOKS)

# Run all tasks
all: install format lint check-style
