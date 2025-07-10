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
	@echo "Logs are written to: /tmp/buttermilk_<run_id>.log"
	@echo "To find the latest log: ls -la /tmp/buttermilk_*.log | tail -1"
	@echo "Starting server in background..."
	@nohup uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full > /dev/null 2>&1 &
	@echo "Server starting... Check logs with: tail -f /tmp/buttermilk_*.log"

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
