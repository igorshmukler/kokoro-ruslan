"""Device capability utilities."""
import logging

import torch

logger = logging.getLogger(__name__)


def check_mps_mixed_precision_support() -> bool:
    """Check if MPS supports mixed precision training."""
    if not torch.backends.mps.is_available():
        return False

    # Check PyTorch version - MPS mixed precision was added in PyTorch 2.0+
    torch_version = torch.__version__.split('.')
    major, minor = int(torch_version[0]), int(torch_version[1])

    if major < 2:
        logger.info(f"MPS autocast requires PyTorch 2.0+, found {torch.__version__}")
        return False

    # Test if autocast works on MPS
    try:
        device = torch.device('mps')
        x = torch.randn(2, 2, device=device)
        # Try using the autocast API
        with torch.autocast(device_type='mps', dtype=torch.float16):
            y = torch.mm(x, x)
        logger.info("MPS autocast support verified")
        return True
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"MPS autocast not supported: {e}")
        return False
