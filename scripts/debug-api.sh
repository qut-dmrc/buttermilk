#!/bin/bash

ACTION="${1:-status}"

case "$ACTION" in
    "start")
        echo "Starting Buttermilk API server..."
        nohup uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full > /tmp/buttermilk_api.log 2>&1 &
        echo "Server starting in background, check /tmp/buttermilk_api.log for details"
        sleep 3
        curl -s http://localhost:8000/health | jq .
        ;;
    "stop")
        echo "Stopping Buttermilk API server..."
        pkill -f "buttermilk.runner.cli.*api"
        echo "Server stopped"
        ;;
    "restart")
        echo "Restarting Buttermilk API server..."
        pkill -f "buttermilk.runner.cli.*api"
        sleep 2
        nohup uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full > /tmp/buttermilk_api.log 2>&1 &
        echo "Server restarting, checking status in 5 seconds..."
        sleep 5
        curl -s http://localhost:8000/health | jq .
        ;;
    "status"|*)
        echo "Checking Buttermilk API server status..."
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ Server is running"
            curl -s http://localhost:8000/health | jq .
        else
            echo "❌ Server is not responding"
            if pgrep -f "buttermilk.runner.cli.*api" >/dev/null; then
                echo "   Process found but not responding"
            else
                echo "   No server process running"
            fi
        fi
        ;;
esac