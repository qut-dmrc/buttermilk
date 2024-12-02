import logging
import time
import uuid

import google.cloud.logging  # Don't conflict with standard logging
import pytest


DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())


@pytest.fixture(scope="function")
def logger_new(bm):
    logger = bm.setup_logging()
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


def test_info(capsys, logger_new, bm):
    logger_new.info(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert "info" in str.lower(captured.err)

    # sleep 5 seconds to allow the last class to write to the cloud service
    time.sleep(5)
    entries = bm._clients['gcslogging'].list_entries(
        order_by=google.cloud.logging.DESCENDING, max_results=100
    )
    for entry in entries:
        if LOG_TEXT in str(entry.payload):
            return True

    raise IOError(f"Info message not found in log: {LOG_TEXT}")


def test_cloud_loger_debug(bm):
    entries = bm._clients['gcslogging'].list_entries(
        order_by=google.cloud.logging.DESCENDING, max_results=100
    )
    for entry in entries:
        if DEBUG_TEXT in str(entry.payload):
            raise IOError(f"Debug message found in log: {LOG_TEXT}")


def test_zzz01_warning_counts(logger_new):
    logger_new.warning("test: logging appears to be working.")
    counts = {}
    for handlerobj in logger_new.handlers:
        if isinstance(handlerobj, CountsHandler):
            counts = handlerobj.get_counts()
            break

    assert counts["WARNING"] == 1
    assert counts["ERROR"] == 0


def test_zzz02_summary_increment(logger_new):
    logger_new.warning("TEST Warning 1")
    logger_new.warning("TEST Warning 2")
    logger_new.increment_run_summary("Test rows saved", 500)
    summary = logger_new.get_log_summary()
    assert "Test rows saved: 500\n" in summary

    # Check that summary contains errors above (Warning: relies on tests running in alphabetical order.)
    assert "WARNING messages: 2\n" in summary


def test_zzz03_test_log_n(logger_new):
    for i in range(0, 20):
        logger_new.log_every_n(f"test log {i}", level=logging.INFO, n=10)

    counts = {}
    for handlerobj in logger_new.handlers:
        if isinstance(handlerobj, CountsHandler):
            counts = handlerobj.get_counts()
            break

    assert counts["INFO"] == 3

def test_stdout_pause(capsys, logger_new):
    logger_new.stdout_pause()
    captured = capsys.readouterr()
    print('test_stdout_pause')
    captured = capsys.readouterr()
    assert captured.out == ''
