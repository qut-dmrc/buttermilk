import time
import uuid

import google.cloud.logging  # Don't conflict with standard logging
import pytest

from buttermilk import BM

DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())


@pytest.fixture(scope="function")
def logger_new(bm):
    bm.setup_logging()
    logger = bm.logger
    yield logger

    logger.info("Tearing test logger_new down.")


def test_error(capsys, logger_new):
    logger_new.error(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert "error" in str.lower(captured.err)


def test_warning(capsys, logger_new):
    logger_new.warning(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert "warning" in str.lower(captured.err)


def test_debug(capsys, logger_new):
    logger_new.debug(DEBUG_TEXT)
    captured = capsys.readouterr()
    assert DEBUG_TEXT not in captured.err
    assert "debug" not in str.lower(captured.err)


@pytest.fixture
def cloud_logging_client_gcs(bm: BM):
    return google.cloud.logging.Client(project=bm.cfg.clouds[0].project)


def test_info(capsys, cloud_logging_client_gcs, logger_new, bm):
    logger_new.info(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert "info" in str.lower(captured.err)

    # sleep 5 seconds to allow the last class to write to the cloud service
    time.sleep(5)
    entries = cloud_logging_client_gcs.list_entries(
        order_by=google.cloud.logging.DESCENDING,
        max_results=100,
    )
    for entry in entries:
        if LOG_TEXT in str(entry.payload):
            return True

    raise OSError(f"Info message not found in log: {LOG_TEXT}")


def test_cloud_loger_debug(cloud_logging_client_gcs):
    entries = cloud_logging_client_gcs.list_entries(
        order_by=google.cloud.logging.DESCENDING,
        max_results=100,
    )
    for entry in entries:
        if DEBUG_TEXT in str(entry.payload):
            raise OSError(f"Debug message found in log: {LOG_TEXT}")


def test_zzz02_summary_increment(logger_new):
    logger_new.warning("TEST Warning 1")
    logger_new.warning("TEST Warning 2")
    logger_new.increment_run_summary("Test rows saved", 500)
    summary = logger_new.get_log_summary()
    assert "Test rows saved: 500\n" in summary

    # Check that summary contains errors above (Warning: relies on tests running in alphabetical order.)
    assert "WARNING messages: 2\n" in summary
