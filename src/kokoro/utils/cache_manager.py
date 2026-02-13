#!/usr/bin/env python3
"""
Utility to check feature cache status and manage cached features.

Usage:
    python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --status
    python3 -m kokoro.utils.cache_manager --corpus ./ruslan_corpus --clear
"""

import argparse
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cache_status(cache_dir: Path):
    """Get cache statistics"""
    if not cache_dir.exists():
        return {
            'exists': False,
            'num_files': 0,
            'total_size_mb': 0.0
        }

    cache_files = list(cache_dir.glob("*.pt"))
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        'exists': True,
        'num_files': len(cache_files),
        'total_size_mb': total_size / (1024**2),
        'cache_dir': str(cache_dir)
    }


def clear_cache(cache_dir: Path, confirm: bool = True):
    """Clear all cached features"""
    if not cache_dir.exists():
        logger.info("Cache directory does not exist")
        return

    status = get_cache_status(cache_dir)

    if confirm:
        print(f"\nCache directory: {cache_dir}")
        print(f"Cached files: {status['num_files']}")
        print(f"Total size: {status['total_size_mb']:.1f} MB")

        response = input("\nAre you sure you want to delete all cached features? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cache clearing cancelled")
            return

    logger.info(f"Clearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    logger.info("Cache cleared successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Manage feature cache for Kokoro-Ruslan"
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default="./ruslan_corpus",
        help="Path to the corpus directory"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: {corpus}/.feature_cache)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached features"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation when clearing"
    )

    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(args.corpus) / ".feature_cache"

    # Execute command
    if args.status:
        status = get_cache_status(cache_dir)

        print("\n" + "="*60)
        print("FEATURE CACHE STATUS")
        print("="*60)

        if status['exists']:
            print(f"Cache directory: {status['cache_dir']}")
            print(f"Cached files: {status['num_files']}")
            print(f"Total size: {status['total_size_mb']:.1f} MB")
            print(f"Average size per file: {status['total_size_mb']/max(status['num_files'], 1):.2f} MB")
        else:
            print("Cache does not exist yet")
            print(f"Expected location: {cache_dir}")
            print("\nRun 'kokoro-precompute' to create the cache")

        print("="*60)

    elif args.clear:
        clear_cache(cache_dir, confirm=not args.force)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
