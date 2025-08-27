# Default target executed when no arguments are given to make.
all: test

######################
# TESTING AND COVERAGE
######################

# Print default configurations
config:
	uv run python -m buttermilk.runner.cli -c job +flows=[trans,transllm,zot,osb] +run=api verbose=true llms=full

kill_chat:
	@echo "Killing chat (frontend) process..."
	@pkill -SIGTERM -f "node.*frontend/chat.*vite dev" || true
	@sleep 5
	@pkill -SIGKILL -f "node.*frontend/chat.*vite dev" || true

kill_api:
	@echo "Killing API (buttermilk) process..."
	@pkill -SIGTERM -f "python.*buttermilk.runner.cli" || true
	@sleep 5
	@pkill -SIGKILL -f "python.*buttermilk.runner.cli" || true

kill: kill_chat kill_api
	@echo "All Buttermilk processes terminated."

# For production API server ONLY.
api:
	uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full

# Run API server in debug mode. Use this one for development.
debug: 
	@echo "Starting Buttermilk API in debug mode..."
	@echo "Logs are written to: /tmp/buttermilk_<run_id>.log"
	@echo "To find the latest log: ./scripts/mcp_debug/getlog.sh"
	@echo "Starting server in background..."
	@nohup uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=debug verbose=true > /dev/null 2>&1 &
	@echo "Server starting... Latest log file (waiting...):"
	@sleep 5s && ./scripts/mcp_debug/getlog.sh 

build:
	@echo "Building Buttermilk Docker image..."
	@docker build -t buttermilk:latest -t us-central1-docker.pkg.dev/prosocial-443205/reg/buttermilk:latest -f deploy/Dockerfile .
	
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