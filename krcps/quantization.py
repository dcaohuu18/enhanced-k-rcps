from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from .utils import get_loss, register_quantization
import torch
import numpy as np


@register_quantization(name="01_loss_kmedoids_q")
def kmedoids(opt_set, opt_l, opt_u, m, _k, _pnk):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    _ki, _kj = torch.nonzero(m[:, :, _k] == 1, as_tuple=True)
    _k_loss = loss[_ki, _kj].numpy()

    kmedoids = KMedoids(n_clusters=_pnk, init="k-medoids++", max_iter=200)
    kmedoids.fit(_k_loss.reshape(-1,1))

    _pi = _ki[kmedoids.medoid_indices_]
    _pj = _kj[kmedoids.medoid_indices_]
    
    return _pi, _pj


@register_quantization(name="01_loss_uniform_q")
def uniform(opt_set, opt_l, opt_u, m, _k, _pnk):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    _ki, _kj = torch.nonzero(m[:, :, _k] == 1, as_tuple=True)
    _k_loss = loss[_ki, _kj].numpy()

    bins = np.linspace(np.quantile(_k_loss, 1/_pnk), 
                       np.quantile(_k_loss, 1 - 1/_pnk), _pnk-1)
    bins_idx = np.digitize(_k_loss, bins, right=False)

    _pi, _pj = [], []
    for i in range(_pnk):
        bi_loss = _k_loss[bins_idx==i]
        if bi_loss.size > 0:
            bi_rep = np.quantile(
              bi_loss, 0.5, method="closest_observation")
            br_idx = np.argwhere(_k_loss==bi_rep).flatten()[0]
        else: 
            # empty bin 
            # can't happen for the first and last interval
            bi_med = (bins[i-1] + bins[i])/2
            br_idx = np.argmin(np.abs(_k_loss - bi_med))
        _pi.append(_ki[br_idx])
        _pj.append(_kj[br_idx])

    return _pi, _pj


@register_quantization(name="01_loss_agglo_q")
def agglomerative(opt_set, opt_l, opt_u, m, _k, _pnk):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    _ki, _kj = torch.nonzero(m[:, :, _k] == 1, as_tuple=True)
    _k_loss = loss[_ki, _kj].numpy()

    ac = AgglomerativeClustering(n_clusters=_pnk)
    ac.fit(_k_loss.reshape(-1,1))

    _pi, _pj = [], []
    for c in range(_pnk):
        c_loss = _k_loss[ac.labels_==c]
        c_rep = np.quantile(
          c_loss, 0.5, method="closest_observation")
        cr_idx = np.argwhere(_k_loss==c_rep).flatten()[0]
        _pi.append(_ki[cr_idx])
        _pj.append(_kj[cr_idx])

    return _pi, _pj