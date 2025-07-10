# Default target executed when no arguments are given to make.
all: test

######################
# TESTING AND COVERAGE
######################

# Print default configurations
config:
	uv run python -m buttermilk.runner.cli -c job

api:
	uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full

# Run API server in debug mode with output capture for automated testing
debug:
	@echo "Starting Buttermilk API in debug mode..."
	@echo "Output will be captured to: /tmp/buttermilk-debug.log"
	@echo "Tail the log with: tail -f /tmp/buttermilk-debug.log"
	@uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full 2>&1 | tee /tmp/buttermilk-debug.log

# Run unit tests and generate a coverage report.
coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

format:	
	uv run python -m black buttermilk

lint:
	uv run python -m  ruff buttermilk

test tests:
	uv run python -m pytest 

scheduled_tests:
	uv run 	python -m pytest -m scheduled tests
