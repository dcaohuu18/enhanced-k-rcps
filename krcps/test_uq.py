import torch


def test_normalize():
    from .uq import _normalize

    l, u = torch.tensor([-1, -1, -1]), torch.tensor([2, 2, 2])
    l_norm, u_norm = _normalize(l, u)
    expected_l_norm, expected_u_norm = torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1])
    assert torch.all(l_norm == expected_l_norm) and torch.all(u_norm == expected_u_norm)

    q_eps = 1e-07
    l = u = torch.tensor([q_eps, q_eps, q_eps])
    l_norm, u_norm = _normalize(l, u)
    expected_l_norm = expected_u_norm = torch.tensor([0, 0, 0])
    assert torch.all(l_norm == expected_l_norm) and torch.all(u_norm == expected_u_norm)


def test_std():
    from .uq import _std

    m = int(1e05)
    sampled = 0.5 + 0.5 * torch.randn(m)
    I = _std(sampled, dim=0)

    l0, u0 = I(0)
    expected_l0 = expected_u0 = torch.tensor(0.5)
    assert torch.isclose(l0, expected_l0, atol=1e-02) and torch.isclose(
        u0, expected_u0, atol=1e-02
    )

    l, u = I(0.5)
    expected_l, expected_u = torch.tensor(0.25), torch.tensor(0.75)
    assert torch.isclose(l, expected_l, atol=1e-02) and torch.isclose(
        u, expected_u, atol=1e-02
    )

    sampled = 0.5 + 1e-04 * torch.randn(m)
    I = _std(sampled, dim=0)

    l, u = I(10)
    expected_l, expected_u = torch.tensor(0.4), torch.tensor(0.6)
    assert torch.isclose(l, expected_l, atol=1e-02) and torch.isclose(
        u, expected_u, atol=1e-02
    )


def test_naive_sampling_additive():
    from .uq import _naive_sampling_additive

    m = 128
    sampled = torch.arange(start=1, end=m + 1).float() / m

    alpha = 0.10
    I = _naive_sampling_additive(sampled, alpha=alpha, dim=0)

    l0, u0 = I(0)
    expected_l0, expected_u0 = 7, 121
    assert (
        len(sampled[sampled <= l0]) == expected_l0
        and len(sampled[sampled <= u0]) == expected_u0
    )

    l, u = I(1 / m)
    expected_l, expected_u = 6, 122
    assert (
        len(sampled[sampled <= l]) == expected_l
        and len(sampled[sampled <= u]) == expected_u
    )


def test_calibrated_quantile():
    from .uq import _calibrated_quantile

    m = 128
    sampled = torch.arange(start=1, end=m + 1).float() / m

    alpha = 0.10
    I = _calibrated_quantile(sampled, alpha=alpha, dim=0)

    l0, u0 = I(0)
    expected_l0, expected_u0 = 6, 123
    assert (
        len(sampled[sampled <= l0]) == expected_l0
        and len(sampled[sampled <= u0]) == expected_u0
    )

    l, u = I(1 / m)
    expected_l, expected_u = 5, 124
    assert (
        len(sampled[sampled <= l]) == expected_l
        and len(sampled[sampled <= u]) == expected_u
    )


def test_quantile_regression():
    from .uq import _quantile_regression

    x = torch.arange(start=1, end=5) / 10
    l = u = torch.ones_like(x)
    denoised = torch.stack([l, x, u], dim=1)

    I = _quantile_regression(denoised)

    l0, u0 = I(0)
    expected_l0 = expected_u0 = x
    assert torch.all(l0 == expected_l0) and torch.all(u0 == expected_u0)

    l, u = I(0.1)
    expected_l, expected_u = x - 0.1, x + 0.1
    assert torch.all(l == expected_l) and torch.all(u == expected_u)

    l = u = torch.zeros_like(x)
    denoised = torch.stack([l, x, u], dim=1)

    I = _quantile_regression(denoised)

    l, u = I(1)
    expected_l, expected_u = x - 1e-02, x + 1e-02
    assert torch.all(l == expected_l) and torch.all(u == expected_u)


def test_conffusion_multiplicative():
    from .uq import _conffusion_multiplicative

    m = 4
    l, u = 0.4 * torch.ones(m), 0.6 * torch.ones(m)
    denoised = torch.stack([l, torch.empty(m), u], dim=1)

    I = _conffusion_multiplicative(denoised)

    l0, u0 = I(1)
    expected_l0, expected_u0 = denoised[:, 0], denoised[:, 2]
    assert torch.all(l0 == expected_l0) and torch.all(u0 == expected_u0)

    l, u = I(1.2)
    expected_l, expected_u = denoised[:, 0] / 1.2, 1.2 * denoised[:, 2]
    assert torch.all(l == expected_l) and torch.all(u == expected_u)


def test_conffusion_additive():
    from .uq import _conffusion_additive

    m = 4
    l, u = 0.4 * torch.ones(m), 0.6 * torch.ones(m)
    denoised = torch.stack([l, torch.empty(m), u], dim=1)

    I = _conffusion_additive(denoised)

    l0, u0 = I(0)
    expected_l0, expected_u0 = denoised[:, 0], denoised[:, 2]
    assert torch.all(l0 == expected_l0) and torch.all(u0 == expected_u0)

    l, u = I(0.1)
    expected_l, expected_u = denoised[:, 0] - 0.1, denoised[:, 2] + 0.1
    assert torch.all(l == expected_l) and torch.all(u == expected_u)
