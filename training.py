#!/usr/bin/env python3
"""
Kokoro Text-To-Speech Language Model Training Script for Russian (Ruslan Corpus)
Main entry point for training
"""

import os
import torch
import logging

from trainer import KokoroTrainer
from cli import parse_arguments, create_config_from_args

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Check MPS availability
    if torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) is available and will be used for acceleration")
    else:
        logger.warning("MPS is not available, falling back to CPU")

    # Create configuration from CLI arguments
    config = create_config_from_args(args)

    # Validate directories
    if not os.path.exists(config.data_dir):
        logger.error(f"Corpus directory not found: {config.data_dir}")
        logger.info("Please ensure the corpus is available in the specified directory")
        return

    logger.info(f"Using corpus directory: {config.data_dir}")
    logger.info(f"Using output directory: {config.output_dir}")

    if config.resume_checkpoint:
        logger.info(f"Resume mode: {config.resume_checkpoint}")

    # Initialize trainer and start training
    trainer = KokoroTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
