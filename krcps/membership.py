import numpy as np
import torch
from skimage.filters import (
  threshold_multiotsu, 
  threshold_isodata, 
  threshold_li,
  threshold_local,
  threshold_triangle,
  threshold_yen)
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from itertools import product
from .utils import get_loss, register_membership


@register_membership(name="01_loss_quantile_m")
def _01_loss_quantile(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    q = torch.quantile(loss.view(-1), torch.arange(0, 1, 1 / k)[1:]).unique()
    k = len(q) + 1

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _q in enumerate(reversed(q)):
        m[loss <= _q] = k - (i + 2)

    # one-hot encoding:
    qcoords = []
    for _k in range(k):
        qcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(qcoords) == len(q) + 1 == k
    assert all([len(_q[0]) == len(_q[1]) for _q in qcoords])
    assert sum([len(_q[0]) for _q in qcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _q in enumerate(qcoords):
        nk[_k] = len(_q[0])
        m[_q[0], _q[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_softquant_m")
def _01_loss_softquant(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss_vec = loss_fn(opt_set, opt_l, opt_u, reduction="none")
    entry_loss = torch.mean(loss_vec, dim=0)

    q = torch.quantile(entry_loss.view(-1), 
                       torch.arange(0, 1, 1 / k)[1:]).unique()
    assert len(q) + 1 == k

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for i, j in product(range(m.size(0)), range(m.size(1))):
        for _k in range(k):
            lq = q[_k-1] if _k>0 else -np.inf
            uq = q[_k] if _k<k-1 else np.inf
            m[i, j, _k] = torch.sum(
                (lq < loss_vec[:, i, j]) & (loss_vec[:, i, j] <= uq))
    # normalize:
    m /= opt_set.size(0)
    # nk gives the number of members per group:
    nk = m.sum(1).sum(0).numpy()
    return k, nk, m


@register_membership(name="01_loss_otsu_m")
def _01_loss_otsu(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_multiotsu(loss.numpy(), classes=k)
    k = len(t) + 1

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _t in enumerate(reversed(t)):
        m[loss <= _t] = k - (i + 2)

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == len(t) + 1 == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_isodata_m")
def _01_loss_isodata(opt_set, opt_l, opt_u):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_isodata(loss.numpy())
    try:
      k = len(t) + 1
    except TypeError:
      t = np.array([t])
      k = 2

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _t in enumerate(reversed(t)):
        m[loss <= _t] = k - (i + 2)

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_li_m")
def _01_loss_li(opt_set, opt_l, opt_u):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_li(loss.numpy())
    k = 2

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    m[loss <= t] = 0

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_local_m")
def _01_loss_local(opt_set, opt_l, opt_u):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_local(loss.numpy())
    m = (loss.numpy() > t).astype(int)
    m = torch.tensor(m)
    k = 2

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_triangle_m")
def _01_loss_triangle(opt_set, opt_l, opt_u):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_triangle(loss.numpy())
    k = 2

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    m[loss <= t] = 0

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_yen_m")
def _01_loss_yen(opt_set, opt_l, opt_u):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_yen(loss.numpy())
    k = 2

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    m[loss <= t] = 0

    # one-hot encoding:
    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_kmeans_m")
def _01_loss_kmeans(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    flat_loss = torch.flatten(loss)
    km = KMeans(n_clusters=k).fit(flat_loss.numpy().reshape(-1,1))
    m = torch.tensor(km.labels_)
    m = torch.reshape(m, loss.size())

    # one-hot encoding:
    ccoords = []
    for _k in range(k):
        ccoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(ccoords) == k
    assert all([len(_c[0]) == len(_c[1]) for _c in ccoords])
    assert sum([len(_c[0]) for _c in ccoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(ccoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_slic_m")
def _01_loss_slic(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    m = slic(loss.numpy(), n_segments=k, start_label=0)
    k = len(np.unique(m))
    m = torch.tensor(m)

    # one-hot encoding:
    scoords = []
    for _k in range(k):
        scoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(scoords) == k
    assert all([len(_s[0]) == len(_s[1]) for _s in scoords])
    assert sum([len(_s[0]) for _s in scoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(scoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1

    return k, nk, m