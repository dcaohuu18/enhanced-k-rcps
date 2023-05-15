import numpy as np


def test_hoeffding_bound():
    from .bounds import _hoeffding_bound

    n, delta, loss = 100, 0.1, 0.5
    bound = _hoeffding_bound(n, delta, loss)
    expected_bound = 0.5 + np.sqrt(np.log(10) / 200)
    assert np.isclose(bound, expected_bound)


def test_hoeffding_plus():
    from .bounds import _hoeffding_plus

    r = loss = 0.5
    n = 100
    hoeffding_plus = _hoeffding_plus(r, loss, n)
    expected_hoeffding_plus = 0.0
    assert np.isclose(hoeffding_plus, expected_hoeffding_plus)


def test_bentkus_plus():
    from .bounds import _bentkus_plus

    r, n = 0.5, 2
    loss = 1 / n
    bentkus_plus = _bentkus_plus(r, loss, n)
    expected_bentkus_plus = np.log(3 * 0.5**2) + 1
    assert np.isclose(bentkus_plus, expected_bentkus_plus)


def test_hoeffing_bentkus_bound():
    from .bounds import _hoeffding_bentkus_bound
