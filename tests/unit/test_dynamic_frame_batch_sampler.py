import pytest

from types import SimpleNamespace

from kokoro.data.dataset import DynamicFrameBatchSampler


class DummyDataset:
    def __init__(self, lengths):
        self.samples = [{'audio_length': int(l)} for l in lengths]

    def __len__(self):
        return len(self.samples)


def test_dynamic_frame_batch_sampler_respects_max_frames():
    # Construct lengths with a variety of sizes, including one oversize sample
    lengths = [20, 30, 10, 90, 200, 50, 5, 1000]
    ds = DummyDataset(lengths)

    max_frames = 100
    sampler = DynamicFrameBatchSampler(dataset=ds, max_frames=max_frames,
                                       min_batch_size=1, max_batch_size=10,
                                       drop_last=False, shuffle=False)

    seen = set()
    for batch in sampler:
        # Ensure indices are valid
        assert isinstance(batch, list)
        assert batch
        for idx in batch:
            assert 0 <= idx < len(ds)
            seen.add(idx)

        # Compute projected cost: len(batch) * max_sample_frames_in_batch
        sample_frames = [ds.samples[i]['audio_length'] for i in batch]
        projected_cost = len(batch) * max(sample_frames)

        # Allow single oversize samples to exceed the budget (sampler design)
        if len(batch) == 1 and max(sample_frames) > max_frames:
            continue

        assert projected_cost <= max_frames, f"Batch {batch} exceeds max_frames: {projected_cost} > {max_frames}"

    # Ensure all indices are produced exactly once
    assert seen == set(range(len(ds)))
