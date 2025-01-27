# Default target executed when no arguments are given to make.
all: test

######################
# TESTING AND COVERAGE
######################


# Run unit tests and generate a coverage report.
coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

format:	
	./venv/bin/python -m black buttermilk

lint:
	./venv/bin/python -m  ruff buttermilk

test tests:
	./venv/bin/python -m pytest 

scheduled_tests:
	./venv/bin/python -m pytest -m scheduled tests
