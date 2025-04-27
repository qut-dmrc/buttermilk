"""
Utility to silence noisy logs from various libraries.
This can be imported and used directly without waiting for the main BM setup_logging process.
"""

import logging
import sys

def silence_task_logs():
    """
    Silence the noisy logging messages from task execution, fsspec, and autogen_core.
    This is especially useful for quieting the console output from asyncio tasks.
    """
    # Silence asyncio task execution logs
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("asyncio.tasks").setLevel(logging.ERROR)
    logging.getLogger("Task").setLevel(logging.ERROR)

    # Silence fsspec logs 
    logging.getLogger("fsspec").setLevel(logging.ERROR)
    logging.getLogger("fsspec.asyn").setLevel(logging.ERROR)
    
    # Silence autogen_core logs
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    logging.getLogger("autogen_core._single_threaded_agent_runtime").setLevel(logging.ERROR)
    
    # Add a null handler to the root logger if it doesn't have any handlers
    # This prevents "No handlers could be found" warnings
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

# Automatically silence logs when module is imported
silence_task_logs()

if __name__ == "__main__":
    print("Silenced noisy task execution logs from asyncio, fsspec, and autogen_core.")
