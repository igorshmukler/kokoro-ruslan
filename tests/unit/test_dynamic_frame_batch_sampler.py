import pytest
import random

from kokoro.data.dataset import DynamicFrameBatchSampler


class DummyDataset:
    def __init__(self, lengths):
        self.samples = [{'audio_length': int(l)} for l in lengths]

    def __len__(self):
        return len(self.samples)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _batch_cost(ds, batch):
    frames = [ds.samples[i]['audio_length'] for i in batch]
    return max(frames) * len(batch)


def _make_sampler(lengths, *, shuffle=False, max_frames=100, min_bs=1, max_bs=10, seed=0):
    random.seed(seed)
    ds = DummyDataset(lengths)
    return ds, DynamicFrameBatchSampler(
        dataset=ds,
        max_frames=max_frames,
        min_batch_size=min_bs,
        max_batch_size=max_bs,
        drop_last=False,
        shuffle=shuffle,
    )


# ─── original contract tests ──────────────────────────────────────────────────

def test_respects_max_frames():
    """No batch should exceed the frame budget (except single-item oversize)."""
    lengths = [20, 30, 10, 90, 200, 50, 5, 1000]
    ds, sampler = _make_sampler(lengths, max_frames=100)

    seen = set()
    for batch in sampler:
        assert isinstance(batch, list) and batch
        for idx in batch:
            assert 0 <= idx < len(ds)
            seen.add(idx)

        sample_frames = [ds.samples[i]['audio_length'] for i in batch]
        projected_cost = len(batch) * max(sample_frames)
        if len(batch) == 1 and max(sample_frames) > 100:
            continue
        assert projected_cost <= 100, f"cost {projected_cost} > max_frames"

    assert seen == set(range(len(ds)))


def test_all_indices_covered_no_shuffle():
    """Every dataset index appears exactly once across all batches (no shuffle)."""
    lengths = list(range(10, 610, 10))  # 60 samples
    ds, sampler = _make_sampler(lengths, max_frames=500, min_bs=1, max_bs=8)

    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)

    assert sorted(all_indices) == list(range(len(ds)))


def test_all_indices_covered_with_shuffle():
    """Every dataset index appears exactly once when shuffle=True."""
    lengths = list(range(10, 410, 10))  # 40 samples
    ds, sampler = _make_sampler(lengths, shuffle=True, max_frames=500, min_bs=1, max_bs=8, seed=42)

    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)

    assert sorted(all_indices) == list(range(len(ds)))


def test_max_batch_size_respected():
    """No batch should contain more than max_batch_size samples."""
    lengths = [50] * 100
    ds, sampler = _make_sampler(lengths, max_frames=10_000, min_bs=1, max_bs=5)

    for batch in sampler:
        assert len(batch) <= 5


def test_empty_dataset():
    ds, sampler = _make_sampler([], max_frames=100)
    assert list(sampler) == []
    assert len(sampler) == 0


def test_single_sample():
    ds, sampler = _make_sampler([300], max_frames=100)
    batches = list(sampler)
    assert len(batches) == 1
    assert batches[0] == [0]


def test_len_matches_iteration():
    """__len__ must match the number of batches actually yielded."""
    lengths = list(range(5, 305, 5))  # 60 samples
    ds, sampler = _make_sampler(lengths, max_frames=300, min_bs=1, max_bs=6)
    assert len(sampler) == len(list(sampler))


# ─── stripe-interleaving tests ───────────────────────────────────────────────

def test_shuffle_rebuilds_batches_each_epoch():
    """Iterating twice with shuffle=True should (almost always) produce different orders."""
    lengths = list(range(10, 510, 10))  # 50 varied lengths
    random.seed(99)
    ds = DummyDataset(lengths)
    sampler = DynamicFrameBatchSampler(ds, max_frames=300, min_batch_size=1,
                                       max_batch_size=6, drop_last=False, shuffle=True)

    epoch1 = [tuple(b) for b in sampler]
    epoch2 = [tuple(b) for b in sampler]
    # With 50 samples it is astronomically unlikely both epochs produce the same order
    assert epoch1 != epoch2, "shuffle=True should produce different batch orders across epochs"


def test_stripe_interleave_heavy_batches_are_spread():
    """
    With shuffle=True the stripe algorithm must ensure the gap between the two
    heaviest batches is at least floor(sqrt(N)) - 1 positions (they end up in
    stripe 0, which is placed every n_stripes steps).
    """
    # 100 samples: 90 short (frames=10) + 10 long (frames=900).
    # The 10 long samples will each form a single-item batch (cost=900).
    # With max_frames=500 each long sample is its own batch.
    short = [10] * 90
    long_ = [900] * 10
    lengths = short + long_
    # max_batch_size=1 forces every sample into its own batch → n=100 batches,
    # n_stripes = int(sqrt(100)) = 10, n_heavy = 10 (= number of long samples).
    # The anchor-first algorithm then places them at stride 10, giving min_gap=10.
    random.seed(7)
    ds = DummyDataset(lengths)
    sampler = DynamicFrameBatchSampler(ds, max_frames=500, min_batch_size=1,
                                       max_batch_size=1, drop_last=False, shuffle=True)

    batches = list(sampler)
    n = len(batches)
    n_stripes = max(2, int(n ** 0.5))

    # Identify positions of the heaviest batches (cost >= 900)
    heavy_positions = [
        i for i, b in enumerate(batches)
        if _batch_cost(ds, b) >= 900
    ]
    assert len(heavy_positions) == 10, "all 10 long samples should appear"

    # Each pair of consecutive heavy positions should be separated by >= n_stripes - 1
    heavy_positions.sort()
    min_gap = min(
        heavy_positions[i + 1] - heavy_positions[i]
        for i in range(len(heavy_positions) - 1)
    )
    assert min_gap >= n_stripes - 1, (
        f"Heavy batches too close: min gap={min_gap}, expected >= {n_stripes - 1}. "
        f"positions={heavy_positions}"
    )


def test_stripe_interleave_no_consecutive_heavy_batches():
    """
    The two absolutely heaviest batches must never be adjacent (gap >= 2)
    after stripe interleaving, for any random seed in a range.
    """
    short = [10] * 80
    long_ = [800] * 4
    lengths = short + long_

    for seed in range(20):
        random.seed(seed)
        ds = DummyDataset(lengths)
        sampler = DynamicFrameBatchSampler(ds, max_frames=400, min_batch_size=1,
                                           max_batch_size=6, drop_last=False, shuffle=True)
        batches = list(sampler)

        heavy_positions = sorted(
            i for i, b in enumerate(batches)
            if _batch_cost(ds, b) >= 800
        )
        for i in range(len(heavy_positions) - 1):
            gap = heavy_positions[i + 1] - heavy_positions[i]
            assert gap >= 2, (
                f"seed={seed}: heavy batches adjacent at positions "
                f"{heavy_positions[i]} and {heavy_positions[i+1]}"
            )


def test_stripe_all_indices_still_covered():
    """Stripe interleaving must not drop or duplicate any indices."""
    short = [10] * 60
    long_ = [600] * 6
    lengths = short + long_
    random.seed(13)
    ds = DummyDataset(lengths)
    sampler = DynamicFrameBatchSampler(ds, max_frames=400, min_batch_size=1,
                                       max_batch_size=6, drop_last=False, shuffle=True)

    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)

    assert sorted(all_indices) == list(range(len(ds)))


def test_no_shuffle_unaffected_by_stripe_logic():
    """When shuffle=False the stripe path is not taken; output is stable across calls."""
    lengths = list(range(10, 310, 10))
    ds, sampler = _make_sampler(lengths, shuffle=False, max_frames=300, min_bs=1, max_bs=6)

    epoch1 = [list(b) for b in sampler]
    epoch2 = [list(b) for b in sampler]
    assert epoch1 == epoch2, "shuffle=False should produce identical order each epoch"


def test_stripe_cost_ordering_descending():
    """
    After stripe interleaving the first batch of each stripe should have
    cost >= the first batch of the next stripe (costs are sorted descending
    before stripe assignment).
    """
    lengths = list(range(5, 505, 5))  # 100 varied-length samples
    random.seed(3)
    ds = DummyDataset(lengths)
    sampler = DynamicFrameBatchSampler(ds, max_frames=1000, min_batch_size=1,
                                       max_batch_size=8, drop_last=False, shuffle=True)

    batches = list(sampler)
    n = len(batches)
    n_stripes = max(2, int(n ** 0.5))

    # Collect costs at interleaved positions
    # Position 0 = stripe 0's first batch (heaviest stripe)
    # Position 1 = stripe 1's first batch, etc.
    # The cost at position 0 should be >= position n_stripes (next stripe-0 batch)
    # is not guaranteed to hold strictly after within-stripe shuffle, but the
    # MAX cost in the first n_stripes positions should be >= max in positions n_stripes..2*n_stripes
    first_block = [_batch_cost(ds, batches[i]) for i in range(min(n_stripes, n))]
    second_block = [_batch_cost(ds, batches[i]) for i in range(n_stripes, min(2 * n_stripes, n))]
    if second_block:
        assert max(first_block) >= max(second_block), (
            "Stripe 0 should contain heavier batches than later stripes"
        )

