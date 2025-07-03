#!/usr/bin/env python3
"""
Emergency manual sync script for ChromaDB remote storage.

Usage:
    python manual_chromadb_sync.py --local ./data/chromadb --remote gs://bucket/chromadb
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time


def check_local_chromadb(local_path: Path) -> bool:
    """Check if local ChromaDB exists and is valid."""
    if not local_path.exists():
        print(f"❌ Local path does not exist: {local_path}")
        return False
        
    sqlite_file = local_path / "chroma.sqlite3"
    if not sqlite_file.exists():
        print(f"❌ No ChromaDB database found at: {sqlite_file}")
        return False
        
    # Get size and modification time
    size_mb = sqlite_file.stat().st_size / (1024 * 1024)
    mtime = time.ctime(sqlite_file.stat().st_mtime)
    
    print(f"✅ Found ChromaDB database:")
    print(f"   Path: {sqlite_file}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Modified: {mtime}")
    
    return True


def check_remote_exists(remote_path: str) -> bool:
    """Check if remote path exists."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", remote_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"⚠️  Remote path exists: {remote_path}")
            
            # Check for sqlite file
            sqlite_check = subprocess.run(
                ["gsutil", "ls", f"{remote_path}/chroma.sqlite3"],
                capture_output=True,
                text=True
            )
            
            if sqlite_check.returncode == 0:
                print("⚠️  Remote ChromaDB database already exists!")
                print("   This will be OVERWRITTEN. Press Ctrl+C to cancel.")
                input("   Press Enter to continue...")
                
        return True
    except Exception as e:
        print(f"❌ Error checking remote: {e}")
        return False


def sync_to_remote(local_path: Path, remote_path: str, dry_run: bool = False) -> bool:
    """Sync local ChromaDB to remote storage."""
    print(f"\n🔄 Syncing ChromaDB to remote storage...")
    print(f"   From: {local_path}")
    print(f"   To:   {remote_path}")
    
    # Build rsync command
    cmd = [
        "gsutil", "-m",  # Multithreaded
        "rsync", "-r",  # Recursive
        "-d",           # Delete remote files not in local
    ]
    
    if dry_run:
        cmd.append("-n")  # Dry run
        print("\n🧪 DRY RUN - No files will be uploaded")
    
    cmd.extend([str(local_path), remote_path])
    
    print(f"\n📡 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Sync completed successfully!")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        else:
            print(f"\n❌ Sync failed with code: {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"\n❌ Sync failed: {e}")
        return False


def verify_remote(remote_path: str) -> bool:
    """Verify remote upload succeeded."""
    print(f"\n🔍 Verifying remote upload...")
    
    try:
        # Check for key files
        files_to_check = [
            "chroma.sqlite3",
            "chroma.sqlite3-shm",
            "chroma.sqlite3-wal"
        ]
        
        for file in files_to_check:
            result = subprocess.run(
                ["gsutil", "ls", "-l", f"{remote_path}/{file}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                # Parse size from output
                parts = result.stdout.strip().split()
                if len(parts) >= 2:
                    size = int(parts[0])
                    size_mb = size / (1024 * 1024)
                    print(f"   ✅ {file}: {size_mb:.2f} MB")
            else:
                print(f"   ⚠️  {file}: Not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def create_backup(local_path: Path) -> bool:
    """Create local backup before sync."""
    backup_path = local_path.parent / f"{local_path.name}_backup_{int(time.time())}"
    
    print(f"\n💾 Creating local backup...")
    print(f"   From: {local_path}")
    print(f"   To:   {backup_path}")
    
    try:
        import shutil
        shutil.copytree(local_path, backup_path)
        print("✅ Backup created successfully")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Manual sync tool for ChromaDB remote storage"
    )
    parser.add_argument(
        "--local",
        type=Path,
        required=True,
        help="Local ChromaDB directory path"
    )
    parser.add_argument(
        "--remote",
        type=str,
        required=True,
        help="Remote storage path (e.g., gs://bucket/chromadb)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip local backup creation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync without confirmation"
    )
    
    args = parser.parse_args()
    
    print("🔧 ChromaDB Manual Sync Tool")
    print("=" * 50)
    
    # Validate local path
    if not check_local_chromadb(args.local):
        sys.exit(1)
    
    # Check remote
    check_remote_exists(args.remote)
    
    # Create backup unless skipped
    if not args.no_backup and not args.dry_run:
        if not create_backup(args.local):
            if not args.force:
                print("\n⚠️  Backup failed. Continue anyway? (y/N)")
                if input().lower() != 'y':
                    print("Aborted.")
                    sys.exit(1)
    
    # Perform sync
    if sync_to_remote(args.local, args.remote, args.dry_run):
        if not args.dry_run:
            # Verify upload
            verify_remote(args.remote)
            
            print("\n✅ SUCCESS: ChromaDB synced to remote storage!")
            print(f"   Remote: {args.remote}")
            print("\n📝 Next steps:")
            print("   1. Test remote access with your application")
            print("   2. Consider setting up automated sync")
            print("   3. Update to latest Buttermilk version for auto-sync fix")
    else:
        print("\n❌ FAILED: Sync did not complete successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()