# How to Trust Your Diffusion Model:<br /> A Convex Optimization Approach to Conformal Risk Control

[![test](https://github.com/Sulam-Group/k-rcps/actions/workflows/test.yml/badge.svg)](https://dl.circleci.com/status-badge/redirect/gh/Sulam-Group/k-rcps/tree/main)
[![codecov](https://codecov.io/gh/Sulam-Group/k-rcps/branch/main/graph/badge.svg?token=PBTV5HYXKR)](https://codecov.io/gh/Sulam-Group/k-rcps)

This is the official implementation of the paper [*How To Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control*](https://arxiv.org/abs/2302.03791)

by [Jacopo Teneggi](https://jacopoteneggi.github.io), Matt Tivnan, J Webster Stayman, and [Jeremias Sulam](https://sites.google.com/view/jsulam).

---

$K$-RCPS is a high-dimensional extension of the [Risk Controlling Prediction Sets (RCPS)](https://github.com/aangelopoulos/rcps) procedure that provably minimizes the mean interval length.

It is based on $\ell^{\gamma}$: a convex upper-bound to the $01$ loss $\ell^{01}$

<p align="center">
  <img width="460" src="assets/loss.jpg">
</p>

## Demo

The demo is included in the `demo.ipynb` notebook. It showcases how to use the $K$-RCPS calibration procedure on dummy data.

<p align="center">
  <img src="assets/results.gif">
</p>

which reduces the mean interval length compared to RCPS on the same data by $\approx 9$%.

## Usage

Let `cal_x, cal_y` be the calibration set containing $n$ i.i.d. samples. To run $K$-RCPS, first construct the family of nested set predictors, and then conformalize.

```python
from krcps.utils import get_uq, get_calibration

# Compute the entrywise calibrated intervals of `cal_y` with miscoverage level `alpha = 0.10`.
alpha = 0.10
calibrated_quantile_fn = get_uq("calibrated_quantile", alpha=alpha, dim=1)
cal_I = calibrated_quantile_fn(m_cal_y)

# Conformalize the family of nested set predictors `cal_I` with number of dimensions `k = 2`
krcps_fn = get_calibration("k_rcps")
_lambda_k = krcps_fn(
  cal_x, cal_I, 
  "hoeffding_bentkus", 
  epsilon=0.10, 
  delta=0.10, 
  lambda_max=0.5, 
  stepsize=2e-03, 
  k=2, 
  "01_loss_otsu", 
  n_opt=128, 
  prob_size=50
)
```

## How to Extend the Current Implementation

$K$-RCPS can be easily extended with new bounds, notions of uncertainty, and membership functions via `krcps/bounds.py`, `krcps/uq.py`, and `krcps/membership.py` respectively.

## References
```
@article{teneggi2023trust,
  title={How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control},
  author={Teneggi, Jacopo and Tivnan, Matt and Stayman, J Webster and Sulam, Jeremias},
  journal={arXiv preprint arXiv:2302.03791},
  year={2023}
}
```