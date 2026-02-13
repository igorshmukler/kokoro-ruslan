"""
Pytest configuration and fixtures for Kokoro-Ruslan tests
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def device():
    """Provide device for testing"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def sample_config():
    """Provide sample training configuration"""
    from kokoro.training.config import TrainingConfig
    return TrainingConfig()


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for tests"""
    return tmp_path


@pytest.fixture
def dataset(sample_config):
    """Provide dataset for testing (uses subset if full corpus available)"""
    from kokoro.data.dataset import RuslanDataset

    # Check if dataset exists
    corpus_path = Path(sample_config.data_dir)
    if not corpus_path.exists():
        pytest.skip(f"Dataset not found at {corpus_path}. Skipping test.")

    # Load dataset (use a subset for faster testing)
    full_dataset = RuslanDataset(sample_config.data_dir, sample_config)

    # Use first 100 samples for testing (faster)
    test_size = min(100, len(full_dataset.samples))

    # Create subset dataset
    subset_dataset = RuslanDataset(
        sample_config.data_dir,
        sample_config,
        indices=list(range(test_size))
    )

    return subset_dataset
