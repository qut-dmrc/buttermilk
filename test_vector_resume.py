#!/usr/bin/env python3
"""
Test script for enhanced vector database resume functionality.

This script validates the key features implemented for issue #66:
- Deduplication strategies
- Resume capability 
- Batch processing with validation
- Local logging functionality
"""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime

from buttermilk._core.types import Record
from buttermilk.data.vector import ChromaDBEmbeddings, ProcessingResult, BatchProcessingResult


async def test_enhanced_vector_database():
    """Test the enhanced vector database functionality."""
    
    print("🧪 TESTING ENHANCED VECTOR DATABASE")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Test directory: {temp_path}")
        
        # Create test vectorstore with new enhanced configuration
        vectorstore = ChromaDBEmbeddings(
            persist_directory=str(temp_path / "chromadb"),
            collection_name="test_collection",
            embedding_model="gemini-embedding-001",
            dimensionality=3072,
            deduplication_strategy="both",  # Test all features
            sync_batch_size=3,  # Small batch for testing
            disable_auto_sync=True  # Disable for testing
        )
        
        # Initialize the vectorstore
        await vectorstore.ensure_cache_initialized()
        
        print(f"✅ Vector store initialized")
        print(f"🔍 Deduplication strategy: {vectorstore.deduplication_strategy}")
        
        # Create test records
        test_records = [
            Record(
                record_id="test-001",
                content="This is the first test document with some content for embedding.",
                metadata={"title": "Test Document 1", "category": "test"}
            ),
            Record(
                record_id="test-002", 
                content="This is the second test document with different content.",
                metadata={"title": "Test Document 2", "category": "test"}
            ),
            Record(
                record_id="test-003",
                content="This is the third test document for comprehensive testing.",
                metadata={"title": "Test Document 3", "category": "test"}
            )
        ]
        
        print(f"\n📚 Created {len(test_records)} test records")
        
        # Test 1: Single record processing
        print(f"\n1️⃣ TESTING SINGLE RECORD PROCESSING")
        print("-" * 40)
        
        result1 = await vectorstore.process_record(
            test_records[0],
            skip_existing=True,
            validate_before_process=True
        )
        
        assert isinstance(result1, ProcessingResult), "Should return ProcessingResult"
        assert result1.status == "processed", f"Expected 'processed', got '{result1.status}'"
        assert result1.chunks_created > 0, "Should create at least one chunk"
        
        print(f"✅ First processing: {result1.status} ({result1.chunks_created} chunks)")
        
        # Test 2: Deduplication (same record should be skipped)
        print(f"\n2️⃣ TESTING DEDUPLICATION")
        print("-" * 40)
        
        result2 = await vectorstore.process_record(
            test_records[0],  # Same record
            skip_existing=True,
            validate_before_process=True
        )
        
        assert isinstance(result2, ProcessingResult), "Should return ProcessingResult"
        assert result2.status == "skipped", f"Expected 'skipped', got '{result2.status}'"
        assert result2.chunks_created == 0, "Should not create new chunks for existing record"
        
        print(f"✅ Duplicate processing: {result2.status} ({result2.reason})")
        
        # Test 3: Force reprocessing
        print(f"\n3️⃣ TESTING FORCE REPROCESSING")
        print("-" * 40)
        
        result3 = await vectorstore.process_record(
            test_records[0],  # Same record but force reprocess
            skip_existing=False,
            force_reprocess=True
        )
        
        assert isinstance(result3, ProcessingResult), "Should return ProcessingResult"
        assert result3.status == "processed", f"Expected 'processed', got '{result3.status}'"
        assert result3.chunks_created > 0, "Should create chunks when forced"
        
        print(f"✅ Force reprocessing: {result3.status} ({result3.chunks_created} chunks)")
        
        # Test 4: Batch processing
        print(f"\n4️⃣ TESTING BATCH PROCESSING")
        print("-" * 40)
        
        batch_result = await vectorstore.process_batch(
            test_records[1:],  # Records 2 and 3 (new)
            mode="safe",
            max_failures=0
        )
        
        assert isinstance(batch_result, BatchProcessingResult), "Should return BatchProcessingResult"
        assert batch_result.total_records == 2, "Should process 2 records"
        assert batch_result.successful_count > 0, "Should process at least one record successfully"
        
        print(f"✅ Batch processing: {batch_result.successful_count} processed, {batch_result.skipped_count} skipped")
        
        # Test 5: Validation-only mode
        print(f"\n5️⃣ TESTING VALIDATION-ONLY MODE")
        print("-" * 40)
        
        validation_result = await vectorstore.process_batch(
            test_records,  # All records (mix of existing and new)
            mode="validate_only"
        )
        
        assert isinstance(validation_result, BatchProcessingResult), "Should return BatchProcessingResult"
        assert validation_result.validation_result is not None, "Should include validation results"
        
        validation = validation_result.validation_result
        total_checked = validation['stats']['would_process'] + validation['stats']['would_skip']
        assert total_checked == len(test_records), f"Should check all {len(test_records)} records"
        
        print(f"✅ Validation: {validation['stats']['would_process']} new, {validation['stats']['would_skip']} existing")
        
        # Test 6: BM Integration and Finalization
        print(f"\n6️⃣ TESTING BM INTEGRATION & FINALIZATION")
        print("-" * 40)
        
        # Finalize to trigger logging
        finalize_success = await vectorstore.finalize_processing()
        assert finalize_success, "Finalization should succeed"
        
        print(f"✅ Finalization successful")
        print(f"   📊 Uses existing BM logging infrastructure")
        
        # Test 7: Collection statistics
        print(f"\n7️⃣ TESTING COLLECTION STATISTICS")
        print("-" * 40)
        
        collection_count = vectorstore.collection.count()
        assert collection_count > 0, "Collection should contain embeddings"
        
        # Test metadata structure
        sample_results = vectorstore.collection.get(limit=1, include=["metadatas"])
        if sample_results["metadatas"]:
            metadata = sample_results["metadatas"][0]
            assert "embedding_model" in metadata, "Should include embedding_model in metadata"
            assert "processing_run_id" in metadata, "Should include processing_run_id in metadata"
            assert "created_timestamp" in metadata, "Should include created_timestamp in metadata"
            assert "content_hash" in metadata, "Should include content_hash in metadata"
            
        print(f"✅ Collection: {collection_count} embeddings with enhanced metadata")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Deduplication: Working correctly")
        print(f"✅ Resume capability: Records skipped/processed appropriately") 
        print(f"✅ Batch processing: Validation and processing working")
        print(f"✅ BM Integration: Uses existing Buttermilk logging infrastructure")
        print(f"✅ Enhanced metadata: Provenance tracking functional")
        
        return True


async def main():
    """Run the test suite."""
    try:
        success = await test_enhanced_vector_database()
        if success:
            print(f"\n🚀 Enhanced vector database is ready for production!")
            return 0
        else:
            print(f"\n❌ Tests failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)