import sys


def test_stop_threshold_aliases(monkeypatch):
    """Verify --stop-threshold sets the correct dest."""
    # Default (no arg)
    monkeypatch.setattr(sys, 'argv', ['prog'])
    from kokoro.cli.cli import parse_arguments
    args = parse_arguments()
    assert abs(args.stop_threshold - 0.1) < 1e-6

    # Original name
    monkeypatch.setattr(sys, 'argv', ['prog', '--stop-threshold', '0.55'])
    args = parse_arguments()
    assert abs(args.stop_threshold - 0.55) < 1e-6
