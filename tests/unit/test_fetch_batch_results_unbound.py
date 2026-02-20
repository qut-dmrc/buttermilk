import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add root to sys.path to import scripts
sys.path.append(os.getcwd())

from scripts.fetch_batch_results import fetch_results


@pytest.mark.anyio
async def test_fetch_results_anypath_accessible() -> None:
    """Test that AnyPath is accessible even if Strategy 1 is skipped."""
    # Build mock_bm before patching to avoid introspecting the uninitialized bm proxy
    mock_bm = MagicMock()
    mock_bm.session_info.save_dir_base = None
    mock_bm.session_info.save_dir = "gs://test-bucket/runs/test-session"

    with (
        patch("scripts.fetch_batch_results.init_async"),
        patch("scripts.fetch_batch_results.bm", new=mock_bm),
        patch("scripts.fetch_batch_results.AnyPath") as mock_anypath,
    ):
        # Mock AnyPath behavior
        mock_path = MagicMock()
        mock_anypath.return_value = mock_path
        mock_path.exists.return_value = False
        mock_path.glob.return_value = []

        # We expect it to exit with 1 because it won't find the manifest,
        # but it SHOULD NOT raise UnboundLocalError/NameError
        with pytest.raises(SystemExit) as excinfo:
            await fetch_results("test_job_id")

        assert excinfo.value.code == 1
        # Strategy 2 should have been attempted
        mock_anypath.assert_called()
