#!/usr/bin/env python3
"""
Test script to verify ChromaDB sync fix works correctly.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from buttermilk.data.vector import ChromaDBEmbeddings


async def test_remote_path_preservation():
    """Test that original remote path is preserved after initialization."""
    print("🧪 Testing remote path preservation...")
    
    remote_path = "gs://test-bucket/chromadb"
    
    # Mock the cache management to avoid actual GCS calls
    with patch('buttermilk.data.vector.ensure_chromadb_cache') as mock_ensure:
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_ensure.return_value = Path(temp_dir)
            
            embeddings = ChromaDBEmbeddings(
                persist_directory=remote_path,
                collection_name="test"
            )
            
            # Before initialization
            assert embeddings.persist_directory == remote_path
            assert embeddings._original_remote_path is None
            
            await embeddings.ensure_cache_initialized()
            
            # After initialization
            assert embeddings._original_remote_path == remote_path, f"Expected {remote_path}, got {embeddings._original_remote_path}"
            assert embeddings.persist_directory == temp_dir, f"Expected {temp_dir}, got {embeddings.persist_directory}"
            
            print("✅ Remote path preservation test passed!")


async def test_sync_path_detection():
    """Test that sync correctly detects remote storage."""
    print("🧪 Testing sync path detection...")
    
    remote_path = "gs://test-bucket/chromadb"
    
    with patch('buttermilk.data.vector.ensure_chromadb_cache') as mock_ensure:
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_ensure.return_value = Path(temp_dir)
            
            # Create fake ChromaDB files
            cache_path = Path(temp_dir)
            (cache_path / "chroma.sqlite3").touch()
            
            with patch('buttermilk.data.vector.upload_chromadb_cache') as mock_upload:
                mock_upload.return_value = None
                
                embeddings = ChromaDBEmbeddings(
                    persist_directory=remote_path,
                    collection_name="test"
                )
                
                await embeddings.ensure_cache_initialized()
                
                # Test that sync now uses the correct paths
                await embeddings._sync_local_changes_to_remote()
                
                # Verify upload was called with correct paths
                mock_upload.assert_called_once_with(
                    str(cache_path),  # Local cache path
                    remote_path       # Original remote path
                )
                
                print("✅ Sync path detection test passed!")


async def test_manual_sync():
    """Test manual sync functionality."""
    print("🧪 Testing manual sync...")
    
    remote_path = "gs://test-bucket/chromadb"
    
    with patch('buttermilk.data.vector.ensure_chromadb_cache') as mock_ensure:
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_ensure.return_value = Path(temp_dir)
            
            # Create fake ChromaDB files
            cache_path = Path(temp_dir)
            (cache_path / "chroma.sqlite3").touch()
            
            with patch('buttermilk.data.vector.upload_chromadb_cache') as mock_upload:
                mock_upload.return_value = None
                
                embeddings = ChromaDBEmbeddings(
                    persist_directory=remote_path,
                    collection_name="test"
                )
                
                await embeddings.ensure_cache_initialized()
                
                # Test manual sync
                success = await embeddings.sync_to_remote(force=True)
                
                assert success, "Manual sync should return True on success"
                
                # Verify upload was called
                mock_upload.assert_called()
                
                print("✅ Manual sync test passed!")


async def test_local_storage_no_sync():
    """Test that local storage doesn't attempt sync."""
    print("🧪 Testing local storage (no sync)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings = ChromaDBEmbeddings(
            persist_directory=temp_dir,  # Local path
            collection_name="test"
        )
        
        await embeddings.ensure_cache_initialized()
        
        # Should not have remote path
        assert embeddings._original_remote_path is None
        assert embeddings.persist_directory == temp_dir
        
        # Manual sync should return True (no-op)
        success = await embeddings.sync_to_remote()
        assert success
        
        print("✅ Local storage test passed!")


async def main():
    """Run all tests."""
    print("🔧 ChromaDB Sync Fix - Test Suite")
    print("=" * 50)
    
    try:
        await test_remote_path_preservation()
        await test_sync_path_detection()
        await test_manual_sync()
        await test_local_storage_no_sync()
        
        print("\n🎉 All tests passed! ChromaDB sync fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)