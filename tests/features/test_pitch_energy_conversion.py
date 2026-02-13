"""
Test pitch/energy conversion from mel-frame level to phoneme level
"""
import torch


def average_pitch_energy_by_duration(values, durations, phoneme_lengths):
    """
    Average frame-level values (pitch/energy) to phoneme-level using durations

    Args:
        values: Frame-level values (batch, mel_frames)
        durations: Phoneme durations (batch, phonemes)
        phoneme_lengths: Actual phoneme lengths (batch,)

    Returns:
        Phoneme-level averaged values (batch, phonemes)
    """
    batch_size = durations.shape[0]
    num_phonemes = durations.shape[1]
    device = durations.device

    # Create output tensor
    phoneme_values = torch.zeros(batch_size, num_phonemes, device=device)

    for b in range(batch_size):
        curr_durations = durations[b]  # (phonemes,)
        curr_values = values[b]  # (frames,)
        actual_phoneme_len = int(phoneme_lengths[b].item())

        frame_idx = 0

        for p in range(actual_phoneme_len):
            dur = int(curr_durations[p].item())
            if dur > 0 and frame_idx < len(curr_values):
                # Average the values for this phoneme's frames
                end_idx = min(frame_idx + dur, len(curr_values))
                phoneme_values[b, p] = curr_values[frame_idx:end_idx].mean()
                frame_idx = end_idx

    return phoneme_values


def test_conversion():
    """Test the conversion from mel-frame to phoneme level"""
    print("Testing pitch/energy conversion from mel-frame to phoneme level...")

    # Simulate batch with:
    # - Batch size 2
    # - Phoneme sequence lengths: [5, 3]
    # - Phoneme durations: [[10, 5, 8, 3, 2], [15, 12, 8]]  (total frames: 28, 35)

    batch_size = 2
    max_phonemes = 5
    max_frames = 50

    # Create durations (padded)
    durations = torch.tensor([
        [10, 5, 8, 3, 2],  # Total: 28 frames
        [15, 12, 8, 0, 0]   # Total: 35 frames (padded with 0s)
    ], dtype=torch.long)

    phoneme_lengths = torch.tensor([5, 3], dtype=torch.long)

    # Create pitch values at mel-frame level (simulated with sequential numbers)
    pitch_mel = torch.zeros(batch_size, max_frames)
    # First sample: frames 0-27
    pitch_mel[0, :28] = torch.arange(28).float()
    # Second sample: frames 0-34
    pitch_mel[1, :35] = torch.arange(35).float() + 100

    print(f"\nInput shapes:")
    print(f"  pitch_mel: {pitch_mel.shape}")
    print(f"  durations: {durations.shape}")
    print(f"  phoneme_lengths: {phoneme_lengths.shape}")

    # Convert to phoneme level
    pitch_phoneme = average_pitch_energy_by_duration(pitch_mel, durations, phoneme_lengths)

    print(f"\nOutput shape:")
    print(f"  pitch_phoneme: {pitch_phoneme.shape}")

    # Verify results
    print("\nSample 1 (5 phonemes):")
    print(f"  Durations: {durations[0, :5].tolist()}")
    print(f"  Expected phoneme pitch values:")
    print(f"    Phoneme 0 (dur=10): avg(0-9) = {torch.arange(10).float().mean():.2f}")
    print(f"    Phoneme 1 (dur=5): avg(10-14) = {torch.arange(10, 15).float().mean():.2f}")
    print(f"    Phoneme 2 (dur=8): avg(15-22) = {torch.arange(15, 23).float().mean():.2f}")
    print(f"    Phoneme 3 (dur=3): avg(23-25) = {torch.arange(23, 26).float().mean():.2f}")
    print(f"    Phoneme 4 (dur=2): avg(26-27) = {torch.arange(26, 28).float().mean():.2f}")
    print(f"  Actual phoneme pitch values: {pitch_phoneme[0, :5].tolist()}")

    print("\nSample 2 (3 phonemes):")
    print(f"  Durations: {durations[1, :3].tolist()}")
    print(f"  Expected phoneme pitch values:")
    print(f"    Phoneme 0 (dur=15): avg(100-114) = {(torch.arange(15).float() + 100).mean():.2f}")
    print(f"    Phoneme 1 (dur=12): avg(115-126) = {(torch.arange(15, 27).float() + 100).mean():.2f}")
    print(f"    Phoneme 2 (dur=8): avg(127-134) = {(torch.arange(27, 35).float() + 100).mean():.2f}")
    print(f"  Actual phoneme pitch values: {pitch_phoneme[1, :3].tolist()}")

    # Verify shapes match
    assert pitch_phoneme.shape == (batch_size, max_phonemes), \
        f"Expected shape {(batch_size, max_phonemes)}, got {pitch_phoneme.shape}"

    # Verify actual values for sample 1
    expected_0 = torch.arange(10).float().mean()
    expected_1 = torch.arange(10, 15).float().mean()
    assert abs(pitch_phoneme[0, 0].item() - expected_0) < 0.01, \
        f"Phoneme 0 mismatch: {pitch_phoneme[0, 0].item()} vs {expected_0}"
    assert abs(pitch_phoneme[0, 1].item() - expected_1) < 0.01, \
        f"Phoneme 1 mismatch: {pitch_phoneme[0, 1].item()} vs {expected_1}"

    print("\n✅ All tests passed! Conversion works correctly.")
    print("\nThe fix should resolve the tensor size mismatch error:")
    print("  - predicted_pitch/energy: phoneme-level (batch, phonemes)")
    print("  - pitch/energy targets: now converted from mel-level to phoneme-level")
    print("  - Both tensors now match in dimension 1!")


if __name__ == "__main__":
    test_conversion()
