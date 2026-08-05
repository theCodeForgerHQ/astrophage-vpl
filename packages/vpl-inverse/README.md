# vpl-inverse

The inverse problem — doc 05.

| Module | Doc section | Contents |
|---|---|---|
| `parameters` | §2.1 | The Level A control-parameter vector `θ_c`, and the bijections onto an unconstrained `R^8` that an optimiser and a sampler both need |
| `priors` | §2.1, §4 | Log-densities for each prior in the §2.1 table, the transform log-Jacobian that makes them correct in unconstrained space, and the §4.2 physics penalties |
| `likelihood` | §3 | Per-channel log-likelihoods — Poisson, heteroscedastic Gaussian, correlated Gaussian, the OES Poisson/Gaussian switch, the doc 01 IF-6 detection gate, and the §3.3 heavy-tailed variants |

This package depends on `vpl-core` and on nothing else in the workspace. That is a
structural requirement, not a coincidence: doc 05 §7.1 makes a forward/inverse model
mismatch mandatory, and an inverse package that could import `vpl.physics` would make
committing the inverse crime the path of least resistance.
