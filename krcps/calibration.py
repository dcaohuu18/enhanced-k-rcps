import os
from typing import Callable, Iterable, Dict

import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    _set_I,
    _split_idx,
    get_bound,
    get_loss,
    get_membership,
    get_quantization,
    register_calibration,
)


def _rcps(
    rcps_set: torch.Tensor,
    I: Callable,
    loss_name: str,
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: torch.Tensor,
    stepsize: float,
    eta: torch.Tensor = None,
    momen: float = 0.1,
    smooth: float = 0.6
):
    loss_fn = get_loss(loss_name)
    bound_fn = get_bound(bound_name)

    n_rcps = rcps_set.size(0)

    _lambda = lambda_max
    if eta is None:
        eta = torch.ones_like(_lambda)
    prev_eta = eta
    eta_changes = torch.zeros_like(eta)

    loss_vec = loss_fn(rcps_set, *I(_lambda), reduction="none")
    prev_loss_vec = loss_vec.clone()
    if eta.size():
        loss_delta = torch.mean(loss_vec, dim=0) 
    else:
        loss_delta = torch.mean(loss_vec)
    ucb = bound_fn(n_rcps, delta, torch.mean(loss_vec))

    pbar = tqdm(total=epsilon)
    pbar.update(ucb)
    pold = ucb

    while ucb <= epsilon:
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(prev_lambda == 0):
            break

        _lambda -= stepsize * eta
        _lambda = torch.clamp(_lambda, min=0)

        loss_vec = loss_fn(rcps_set, *I(_lambda), reduction="none")
        if eta.size():
            entry_loss = torch.mean(loss_vec, dim=0)
            prev_entry_loss = torch.mean(prev_loss_vec, dim=0)
            loss_delta = smooth*(entry_loss - prev_entry_loss) + (1-smooth)*loss_delta 
        else:
            loss_delta = smooth*(torch.mean(loss_vec) - torch.mean(prev_loss_vec)
                         ) + (1-smooth)*loss_delta 
        loss_delta = torch.clamp(loss_delta, min=0.0)
        eta = torch.clamp(eta - loss_delta, min=0.0)
        prev_ucb = ucb
        ucb = bound_fn(n_rcps, delta, torch.mean(loss_vec))
        eta_changes = prev_eta - eta
        prev_eta = eta
        prev_loss_vec = loss_vec

    _lambda = prev_lambda
    ucb = prev_ucb

    while not torch.all(eta == 0):
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(prev_lambda == 0):
            break

        _lambda -= stepsize * eta
        _lambda = torch.clamp(_lambda, min=0)

        loss_vec = loss_fn(rcps_set, *I(_lambda), reduction="none")
        prev_ucb = ucb
        ucb = bound_fn(n_rcps, delta, torch.mean(loss_vec))

        if ucb > epsilon:
            if eta.size():
                entry_loss = torch.mean(loss_vec, dim=0) 
                eta -= (entry_loss + momen*eta_changes)
            else:
                eta -= (torch.mean(loss_vec) + momen*eta_changes)
            eta = torch.clamp(eta, min=0.0)
            _lambda = prev_lambda
            ucb = prev_ucb

        eta_changes = prev_eta - eta
        prev_eta = eta

    pbar.update(epsilon - pold)
    pbar.close()
    return _lambda


@register_calibration(name="rcps")
def _calibrate_rcps(
    cal_set: torch.Tensor,
    I: Callable[[torch.Tensor], torch.Tensor],
    loss_name: str,
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: float,
    stepsize: float,
):
    lambda_max = torch.tensor(lambda_max)
    _lambda = _rcps(
        cal_set, I, loss_name, bound_name, epsilon, delta, lambda_max, stepsize
    )
    return _lambda


def _gamma_loss_fn(i, offset, q, _lambda):
    i_lambda = i + 2 * _lambda
    inv_i_lambda = cp.multiply(cp.inv_pos(i_lambda), offset)
    loss = 2 * (1 + q) * inv_i_lambda - q
    loss = cp.pos(loss)
    return loss


def _pk(opt_set, opt_I, epsilon, lambda_max, k, membership_name, 
        prob_size, quantization_name):
    n_opt = opt_set.size(0)
    opt_l, opt_u = opt_I(0)

    membership_fn = get_membership(membership_name)
    if k > 0:
        k, nk, m = membership_fn(opt_set, opt_l, opt_u, k)
    else: # membership function without preset k
        k, nk, m = membership_fn(opt_set, opt_l, opt_u)

    d = np.prod(opt_set.size()[-2:])
    # sample d_opt stratified by membership:
    prob_nk = np.round(prob_size / d * nk).astype(int)
    prob_i, prob_j, prob_lambda = [], [], []
    for _k, _pnk in enumerate(prob_nk):
        if quantization_name is not None:
            quantization_fn = get_quantization(quantization_name)
            _pi, _pj = quantization_fn(opt_set, opt_l, opt_u, m, _k, _pnk)
        else:
            # get all the possible coords i,j:
            _ki, _kj = torch.nonzero(m[:, :, _k] != -2, as_tuple=True)
            # sample based on normalized membership degree
            _kidx = np.random.choice(
              d, size=_pnk, replace=False,
              p = (torch.flatten(m[:, :, _k])/torch.sum(m[:, :, _k])).numpy()
            )
            _pi, _pj = _ki[_kidx], _kj[_kidx]
        
        prob_i.extend(_pi)
        prob_j.extend(_pj)
        prob_lambda.extend(_pnk * [_k])

    _lambda = cp.Variable(k)
    q = cp.Parameter(nonneg=True)

    c = (opt_l + opt_u) / 2
    i = opt_u - opt_l
    offset = torch.abs(opt_set - c)
    i_npy, offset_npy = i.numpy(), offset.numpy()
    r_hat = cp.sum(
        _gamma_loss_fn(
            i_npy[:, prob_i, prob_j],
            offset_npy[:, prob_i, prob_j],
            q,
            _lambda[[prob_lambda]],
        )
    ) / (n_opt * np.sum(prob_nk))

    obj = cp.Minimize(cp.sum(cp.multiply(prob_nk, _lambda)))
    constraints = [_lambda >= 0, _lambda <= lambda_max, r_hat <= epsilon]
    pk = cp.Problem(obj, constraints)
    return (pk, q, _lambda), m


def _solve(q, pk, _lambda, gamma):
    q.value = gamma / (1 - gamma)
    if os.path.exists(os.path.expanduser("~/mosek/mosek.lic")):
        pk.solve(
            solver=cp.MOSEK,
            verbose=False,
            warm_start=True,
            mosek_params={"MSK_IPAR_NUM_THREADS": 1},
        )
    else:
        pk.solve(verbose=False, warm_start=True)
    lambda_k, obj = torch.tensor(_lambda.value, dtype=torch.float32), pk.value
    return lambda_k, obj


@register_calibration(name="k_rcps")
def _calibrate_k_rcps(
    cal_set: torch.Tensor,
    I: Callable[[torch.Tensor], torch.Tensor],
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: float,
    stepsize: torch.Tensor,
    membership_k: Dict[str, int],
    n_opt: int,
    prob_size: float,
    gamma: Iterable[float],
    quantization_name: str,
    lambda_agg_fn: Callable = torch.median,
):
    n = cal_set.size(0)
    opt_idx, rcps_idx = _split_idx(n, n_opt)

    opt_set = cal_set[opt_idx]
    rcps_set = cal_set[rcps_idx]

    opt_I = _set_I(I, opt_idx)
    rcps_I = _set_I(I, rcps_idx)

    agg_lambda = torch.zeros(len(membership_k), *cal_set.size()[1:])

    for i, (memf, k) in enumerate(membership_k.items()):
      prob, m = _pk(opt_set, opt_I, epsilon, lambda_max, k, memf, 
                    prob_size, quantization_name)
      pk, q, m_lambda = prob

      sol = [_solve(q, pk, m_lambda, _gamma) for _gamma in tqdm(gamma)]
      sol = sorted(sol, key=lambda x: x[-1])
      lambda_k, _ = sol[0]
      agg_lambda[i] = torch.matmul(m, lambda_k)
    
    agg_lambda = lambda_agg_fn(agg_lambda, dim=0)
    if not torch.is_tensor(agg_lambda):
      agg_lambda = agg_lambda.values
    agg_lambda += lambda_max 

    _lambda = _rcps(
        rcps_set, rcps_I, "vector_01", bound_name, epsilon, delta, agg_lambda, stepsize
    )
    return _lambda
