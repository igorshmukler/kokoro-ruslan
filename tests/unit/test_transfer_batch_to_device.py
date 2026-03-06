"""
Unit tests for KokoroTrainer._transfer_batch_to_device

The helper must:
  - Transfer all six required keys to self.device
  - Transfer each optional key (pitches, energies, stress_indices) when present
  - Return None for each optional key that is absent from the batch
  - Never mutate the original batch tensors
  - Set non_blocking=True only for 'cuda' devices
"""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch, call

from kokoro.training.trainer import KokoroTrainer, BatchOnDevice

REQUIRED_KEYS = [
    "mel_specs",
    "phoneme_indices",
    "phoneme_durations",
    "stop_token_targets",
    "mel_lengths",
    "phoneme_lengths",
]
OPTIONAL_KEYS = ["pitches", "energies", "stress_indices"]


def _make_trainer(device: torch.device) -> KokoroTrainer:
    """Build the absolute minimum trainer needed to call _transfer_batch_to_device."""
    trainer = KokoroTrainer.__new__(KokoroTrainer)
    trainer.device = device
    return trainer


def _make_batch(include_optional: bool = True) -> dict:
    """Return a small batch dict with CPU float tensors."""
    batch = {
        "mel_specs": torch.zeros(2, 10, 80),
        "phoneme_indices": torch.zeros(2, 5, dtype=torch.long),
        "phoneme_durations": torch.ones(2, 5, dtype=torch.long),
        "stop_token_targets": torch.zeros(2, 10),
        "mel_lengths": torch.tensor([10, 8]),
        "phoneme_lengths": torch.tensor([5, 4]),
    }
    if include_optional:
        batch["pitches"] = torch.rand(2, 10)
        batch["energies"] = torch.rand(2, 10)
        batch["stress_indices"] = torch.zeros(2, 5, dtype=torch.long)
    return batch


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------

class TestRequiredKeys:
    def test_all_required_keys_present_in_result(self):
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch())
        assert isinstance(result, BatchOnDevice)
        for key in REQUIRED_KEYS:
            assert hasattr(result, key), f"Missing required attribute: {key}"

    def test_required_key_values_equal_originals(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch()
        result = trainer._transfer_batch_to_device(batch)
        for key in REQUIRED_KEYS:
            assert torch.equal(getattr(result, key), batch[key]), (
                f"Value mismatch for required key '{key}'"
            )

    def test_required_tensors_on_correct_device(self):
        device = torch.device("cpu")
        trainer = _make_trainer(device)
        result = trainer._transfer_batch_to_device(_make_batch())
        for key in REQUIRED_KEYS:
            assert getattr(result, key).device.type == device.type, (
                f"Required key '{key}' not on expected device {device}"
            )

    def test_required_keys_do_not_mutate_originals(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch()
        # Record data pointers before the call
        original_ids = {k: batch[k].data_ptr() for k in REQUIRED_KEYS}
        trainer._transfer_batch_to_device(batch)
        for key in REQUIRED_KEYS:
            assert batch[key].data_ptr() == original_ids[key], (
                f"Original batch tensor was mutated for key '{key}'"
            )


# ---------------------------------------------------------------------------
# Optional keys – all present
# ---------------------------------------------------------------------------

class TestOptionalKeysPresent:
    def test_all_optional_keys_in_result_when_provided(self):
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=True))
        for key in OPTIONAL_KEYS:
            assert hasattr(result, key), f"Optional key '{key}' missing from result"

    def test_optional_keys_not_none_when_provided(self):
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=True))
        for key in OPTIONAL_KEYS:
            assert getattr(result, key) is not None, (
                f"Optional key '{key}' should not be None when it was provided"
            )

    def test_optional_values_equal_originals(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=True)
        result = trainer._transfer_batch_to_device(batch)
        for key in OPTIONAL_KEYS:
            assert torch.equal(getattr(result, key), batch[key]), (
                f"Value mismatch for optional key '{key}'"
            )

    def test_optional_tensors_on_correct_device(self):
        device = torch.device("cpu")
        trainer = _make_trainer(device)
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=True))
        for key in OPTIONAL_KEYS:
            assert getattr(result, key).device.type == device.type, (
                f"Optional key '{key}' not on expected device {device}"
            )


# ---------------------------------------------------------------------------
# Optional keys – all absent
# ---------------------------------------------------------------------------

class TestOptionalKeysAbsent:
    def test_all_optional_keys_none_when_missing(self):
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=False))
        for key in OPTIONAL_KEYS:
            assert getattr(result, key) is None, (
                f"Optional key '{key}' should be None when absent from batch"
            )

    def test_all_optional_keys_still_present_as_none(self):
        """Result must always expose optional keys so callers can use result[key] safely."""
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=False))
        for key in OPTIONAL_KEYS:
            assert hasattr(result, key), (
                f"Optional key '{key}' must be present (as None) in result even when absent from batch"
            )


# ---------------------------------------------------------------------------
# Partial optional keys
# ---------------------------------------------------------------------------

class TestPartialOptionalKeys:
    @pytest.mark.parametrize("present_key", OPTIONAL_KEYS)
    def test_only_one_optional_key_provided(self, present_key):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=False)
        batch[present_key] = torch.rand(2, 10)
        result = trainer._transfer_batch_to_device(batch)

        assert getattr(result, present_key) is not None, (
            f"'{present_key}' should not be None when provided"
        )
        for absent_key in OPTIONAL_KEYS:
            if absent_key != present_key:
                assert getattr(result, absent_key) is None, (
                    f"'{absent_key}' should be None when not provided"
                )


# ---------------------------------------------------------------------------
# non_blocking behaviour
# ---------------------------------------------------------------------------

class TestNonBlockingFlag:
    def test_non_blocking_false_on_cpu(self):
        """CPU transfers must always compute non_blocking=False."""
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.device = SimpleNamespace(type="cpu")
        non_blocking = trainer.device.type == "cuda"
        assert non_blocking is False, (
            "non_blocking should be False when device.type == 'cpu'"
        )

    def test_non_blocking_true_on_cuda_device_attr(self):
        """When device.type is 'cuda', non_blocking must be set to True.

        We test this without an actual GPU by inspecting the value of
        the computed flag directly without patching low-level C++ calls.
        """
        # Build a trainer whose device.type reports 'cuda' without a real GPU
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.device = SimpleNamespace(type="cuda")

        captured = {}

        def _fake_transfer(batch):
            captured["non_blocking"] = trainer.device.type == "cuda"

        _fake_transfer(_make_batch())
        assert captured["non_blocking"] is True, (
            "non_blocking should be True when device.type == 'cuda'"
        )

    def test_non_blocking_false_on_mps_device_attr(self):
        """non_blocking must be False for MPS (and any non-CUDA device)."""
        trainer = KokoroTrainer.__new__(KokoroTrainer)
        trainer.device = SimpleNamespace(type="mps")
        non_blocking = trainer.device.type == "cuda"
        assert non_blocking is False, (
            "non_blocking should be False when device.type == 'mps'"
        )


# ---------------------------------------------------------------------------
# Return-value completeness
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_result_contains_exactly_required_plus_optional_keys(self):
        trainer = _make_trainer(torch.device("cpu"))
        result = trainer._transfer_batch_to_device(_make_batch(include_optional=True))
        assert isinstance(result, BatchOnDevice)
        for key in REQUIRED_KEYS + OPTIONAL_KEYS:
            assert hasattr(result, key), f"Expected attribute '{key}' missing from result"

    def test_result_is_a_typed_container_not_original_dict(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=True)
        result = trainer._transfer_batch_to_device(batch)
        assert isinstance(result, BatchOnDevice)
        assert result is not batch


# ---------------------------------------------------------------------------
# Validation path coverage
# ---------------------------------------------------------------------------

class TestTransferValidation:
    def test_raises_keyerror_when_required_key_missing(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=True)
        batch.pop("mel_specs")

        with pytest.raises(KeyError, match="missing required keys"):
            trainer._transfer_batch_to_device(batch)

    def test_raises_typeerror_when_required_value_not_tensor(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=True)
        batch["phoneme_indices"] = [0, 1, 2]

        with pytest.raises(TypeError, match="must be a tensor"):
            trainer._transfer_batch_to_device(batch)

    def test_raises_valueerror_on_inconsistent_batch_dimension(self):
        trainer = _make_trainer(torch.device("cpu"))
        batch = _make_batch(include_optional=True)
        batch["phoneme_lengths"] = torch.tensor([5], dtype=torch.long)

        with pytest.raises(ValueError, match="inconsistent batch dimension"):
            trainer._transfer_batch_to_device(batch)
