import logging
import os
from logging import getLogger
from pathlib import Path

import structlog
from google.cloud import logging as gcp_logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from rich.logging import RichHandler

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"


# Standard Python logger for console output
logger = getLogger(_LOGGER_NAME)

# Structured logger for JSON output (files and cloud)
struct_logger = structlog.get_logger(_LOGGER_NAME)


# Configure structlog for structured JSON logging
def configure_structlog() -> None:
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            # Add context variables automatically
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp  
            structlog.processors.TimeStamper(fmt="iso"),
            # Output as JSON
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def setup_console_logging(verbose: bool = False) -> None:
    """Set up beautiful console logging with Rich.
    
    Args:
        verbose: If True, shows DEBUG level logs on console
    """
    # Create Rich handler for beautiful console output
    rich_handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    
    # Set console level based on verbosity
    console_level = logging.DEBUG if verbose else logging.INFO
    rich_handler.setLevel(console_level)
    
    # Clear existing handlers and set up clean console logging
    logger.handlers.clear()
    logger.addHandler(rich_handler)
    logger.setLevel(console_level)


def setup_file_logging(run_id: str, verbose: bool = False) -> list[str]:
    """Set up structured JSON logging to files.
    
    Args:
        run_id: Unique run identifier for log file naming
        verbose: If True, creates both INFO and DEBUG log files
        
    Returns:
        List of log file paths created
    """
    # Configure structlog if not already done
    configure_structlog()
    
    log_files = []
    
    # Set up context for this session
    session_id = session_id_var.get()
    agent_id = agent_id_var.get()
    
    if session_id or agent_id:
        structlog.contextvars.bind_contextvars(
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id
        )
    
    # Always create an INFO JSON log file
    info_log_path = Path(f"/tmp/buttermilk_{run_id}_info.jsonl")
    info_handler = logging.FileHandler(info_log_path, mode="w")
    info_handler.setLevel(logging.INFO)
    
    # Use structlog formatter for JSON output with full processing pipeline
    structlog_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            # Add context variables automatically
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Output as JSON
            structlog.processors.JSONRenderer()
        ],
    )
    info_handler.setFormatter(structlog_formatter)
    
    # Add to both standard logger and structlog
    logging.getLogger().addHandler(info_handler)
    log_files.append(str(info_log_path))
    
    # Add debug file logging when verbose is True
    if verbose:
        debug_log_path = Path(f"/tmp/buttermilk_{run_id}_debug.jsonl")
        debug_handler = logging.FileHandler(debug_log_path, mode="w")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(structlog_formatter)
        
        logging.getLogger().addHandler(debug_handler)
        log_files.append(str(debug_log_path))
    
    return log_files


def setup_cloud_logging(logger_cfg, cloud_manager, run_info) -> None:
    """Set up Google Cloud Logging with structured JSON.
    
    Uses the same structlog JSON format as file logging for consistency.
    
    Args:
        logger_cfg: Logger configuration object
        cloud_manager: Cloud manager instance for GCS client access
        run_info: Session run information
    """
    if logger_cfg and logger_cfg.type == "gcp" and cloud_manager:
        try:
            # Ensure structlog is configured
            configure_structlog()
            
            cloud_logging_resource = gcp_logging.Resource(
                type="generic_task",
                labels={
                    "project": logger_cfg.project_id,
                    "location": logger_cfg.location,
                    "namespace": run_info.name,
                    "job": run_info.job,
                    "task_id": run_info.run_id,
                },
            )

            cloud_handler = CloudLoggingHandler(
                client=cloud_manager.gcs_log_client(logger_cfg),
                resource=cloud_logging_resource,
                name=run_info.name,
                labels=run_info.model_dump(include={"run_id", "name", "job", "platform"}),
            )
            cloud_handler.setLevel(logging.INFO)
            
            # Use structlog JSON formatter for consistency with file logging
            structlog_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    # Add context variables automatically
                    structlog.contextvars.merge_contextvars,
                    # Add log level
                    structlog.processors.add_log_level,
                    # Add timestamp
                    structlog.processors.TimeStamper(fmt="iso"),
                    # Output as JSON
                    structlog.processors.JSONRenderer()
                ],
            )
            cloud_handler.setFormatter(structlog_formatter)
            
            # Bind session context for automatic inclusion
            structlog.contextvars.bind_contextvars(
                session_id=run_info.run_id[-12:],  # Last 12 chars for brevity
                run_id=run_info.run_id,
                job=run_info.job,
                project=run_info.name
            )
            
            # Add to root logger so all log messages go to cloud
            logging.getLogger().addHandler(cloud_handler)
            logger.info("Cloud logging handler added")
            
        except Exception as e:
            logger.error(
                "Cloud logging setup failed",
                extra={
                    "error": str(e),
                    "logger_type": logger_cfg.type,
                    "project_id": logger_cfg.project_id,
                    "location": logger_cfg.location
                }
            )


# Add a highlight method to the standard logger for backwards compatibility
def highlight(msg: str, *args, **kwargs) -> None:
    """Log a highlighted message that stands out in console."""
    logger.info(f"✓ {msg}", *args, **kwargs)


# Monkey patch the highlight method onto the logger
logger.highlight = highlight
