import torch

from src.kokoro.training.mps_grad_scaler import MPSGradScaler


def test_mps_grad_scaler_step_and_update():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([p], lr=0.1)
    # set grad manually
    p.grad = torch.tensor([0.5])

    scaler = MPSGradScaler(init_scale=4.0, growth_interval=1)
    assert scaler.get_scale() == 4.0

    ok = scaler.step(optimizer)
    assert ok is True
    # grad was divided by scale before step: 0.5/4 = 0.125
    expected = 1.0 - 0.1 * 0.125
    assert torch.allclose(p.detach(), torch.tensor([expected]))

    # growth_tracker incremented and triggers update
    scaler.update()
    assert scaler.get_scale() == 8.0


def test_mps_grad_scaler_nan_grad_skips_and_backoff():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([p], lr=0.1)
    p.grad = torch.tensor([float('nan')])

    scaler = MPSGradScaler(init_scale=16.0)
    ok = scaler.step(optimizer)
    assert ok is False
    # scale reduced by backoff (0.5)
    assert scaler.get_scale() == 8.0
    # grad should be cleared by optimizer.zero_grad()
    assert p.grad is None or torch.isnan(p.grad).all()


def test_state_dict_and_load():
    s1 = MPSGradScaler(init_scale=2.0)
    s1._growth_tracker = 5
    st = s1.state_dict()

    s2 = MPSGradScaler()
    s2.load_state_dict(st)
    assert s2.get_scale() == 2.0
    assert s2._growth_tracker == 5
