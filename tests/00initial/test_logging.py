import asyncio
import uuid

import pytest

from buttermilk.bm import (  # Buttermilk global instance and logger
    BM,  # Buttermilk global instance and logger
    get_bm,
)

bm = get_bm()

DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())

bm = bm


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


@pytest.mark.anyio
async def test_warning(capsys, logger_new, bm: BM):
    log_text = f"{LOG_TEXT} {uuid.uuid4()}"
    logger_new.warning(log_text)
    captured = capsys.readouterr()
    assert log_text in captured.err
    assert "warning" in str.lower(captured.err)

    # sleep 5 seconds to allow the last class to write to the cloud service
    await asyncio.sleep(5)
    from google.cloud.logging_v2 import DESCENDING

    entries = bm.gcs_log_client.list_entries(
        order_by=DESCENDING,
        max_results=100,
    )
    for entry in entries:
        if log_text in str(entry.payload):
            return True

    raise OSError(f"Warning message not found in log: {log_text}")


def test_debug(capsys, logger_new):
    logger_new.debug(DEBUG_TEXT)
    captured = capsys.readouterr()
    assert DEBUG_TEXT not in captured.err
    assert "debug" not in str.lower(captured.err)


@pytest.fixture
def cloud_logging_client_gcs(bm: BM):
    return bm.gcs_log_client


def test_info(capsys, cloud_logging_client_gcs, logger_new):
    logger_new.info(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert "info" in str.lower(captured.err)


def test_cloud_loger_debug(cloud_logging_client_gcs):
    entries = cloud_logging_client_gcs.list_entries(
        order_by=google.cloud.logging.DESCENDING,
        max_results=100,
    )
    for entry in entries:
        if DEBUG_TEXT in str(entry.payload):
            raise OSError(f"Debug message found in log: {LOG_TEXT}")
