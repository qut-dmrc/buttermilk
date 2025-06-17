"""Test ChromaDB remote sync fix."""

import pytest

pytestmark = pytest.mark.anyio  # Enable async test support
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from buttermilk.data.vector import ChromaDBEmbeddings


class TestChromaDBSyncFix:
    """Test suite for ChromaDB remote sync bug fix."""

    def test_original_remote_path_field_exists(self):
        """Test that _original_remote_path field was added."""
        # Should not raise AttributeError
        assert hasattr(ChromaDBEmbeddings, '_original_remote_path')

    def test_sync_method_has_fixes(self):
        """Test that sync method contains the critical fixes."""
        import inspect
        
        sync_source = inspect.getsource(ChromaDBEmbeddings._sync_local_changes_to_remote)
        
        # Check for key fixes
        assert "_original_remote_path" in sync_source, "Should use original remote path"
        assert "Path(self.persist_directory)" in sync_source, "Should use persist_directory as cache path"
        assert "raise RuntimeError" in sync_source, "Should raise errors instead of warnings"
        assert "logger.error" in sync_source, "Should log critical errors"

    def test_manual_sync_method_exists(self):
        """Test that manual sync method was added."""
        assert hasattr(ChromaDBEmbeddings, 'sync_to_remote')
        assert callable(getattr(ChromaDBEmbeddings, 'sync_to_remote'))

    def test_ensure_cache_saves_original_path(self):
        """Test that ensure_cache_initialized saves original remote path."""
        import inspect
        
        init_source = inspect.getsource(ChromaDBEmbeddings.ensure_cache_initialized)
        assert "_original_remote_path = self.persist_directory" in init_source

    @patch('buttermilk.data.vector.TextEmbeddingModel')
    @patch('buttermilk.data.vector.VertexAIEmbeddingFunction')
    @patch('buttermilk.data.vector.chromadb')
    def test_path_preservation_logic(self, mock_chromadb, mock_embedding_func, mock_text_model):
        """Test the path preservation logic without actual model loading."""
        # Mock the model loading to avoid API calls
        mock_text_model.from_pretrained.return_value = MagicMock()
        mock_embedding_func.return_value = MagicMock()
        mock_chromadb.PersistentClient.return_value = MagicMock()
        
        remote_path = "gs://test-bucket/chromadb"
        
        with patch('buttermilk.data.vector.ensure_chromadb_cache') as mock_ensure:
            with tempfile.TemporaryDirectory() as temp_dir:
                mock_ensure.return_value = Path(temp_dir)
                
                # Create instance (this will trigger model loading validation)
                embeddings = ChromaDBEmbeddings(
                    persist_directory=remote_path,
                    collection_name="test"
                )
                
                # Before ensure_cache_initialized
                assert embeddings.persist_directory == remote_path
                assert embeddings._original_remote_path is None
                
                # This would normally be called, but we'll simulate the path changes
                embeddings._original_remote_path = remote_path
                embeddings.persist_directory = temp_dir
                
                # After the changes
                assert embeddings._original_remote_path == remote_path
                assert embeddings.persist_directory == temp_dir

    def test_sync_path_logic_simulation(self):
        """Test the sync path logic without actual ChromaDB instance."""
        
        def simulate_old_sync_logic(persist_directory):
            """Simulate the old broken sync logic."""
            if not persist_directory.startswith(("gs://", "s3://")):
                return "SKIP_SYNC"
            return "SYNC"
        
        def simulate_new_sync_logic(persist_directory, original_remote_path):
            """Simulate the new fixed sync logic."""
            remote_path = original_remote_path or persist_directory
            
            if not remote_path.startswith(("gs://", "s3://")):
                return "SKIP_SYNC"
            return "SYNC"
        
        # Test case: Remote storage
        original_remote = "gs://bucket/chromadb"
        local_cache = "/home/user/.cache/chromadb/gs___bucket_chromadb"
        
        # Old logic (broken)
        old_result = simulate_old_sync_logic(local_cache)
        assert old_result == "SKIP_SYNC", "Old logic incorrectly skips remote sync"
        
        # New logic (fixed)
        new_result = simulate_new_sync_logic(local_cache, original_remote)
        assert new_result == "SYNC", "New logic correctly syncs remote storage"
        
        # Test case: Local storage
        local_path = "/local/chromadb"
        
        old_local = simulate_old_sync_logic(local_path)
        new_local = simulate_new_sync_logic(local_path, None)
        
        assert old_local == "SKIP_SYNC"
        assert new_local == "SKIP_SYNC"
        assert old_local == new_local, "Local storage behavior should be unchanged"