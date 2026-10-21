# Comparing Foster's inverse variants

This branch adds a comparison layer around the original
`fosters_inverse_sss.py`, which is **left unchanged**.

Two priors are evaluated with the same SSS basis, the same sensor data, the
same sensor noise covariance `N`, and the same affine term:

| variant | moment prior `A` | normalization |
|---|---|---|
| `xan` | `np.cov(x_hat)` (raw covariance) | none |
| `mne` | `PSD(np.cov(x_hat) - pS_in @ N @ pS_in.T)` | `A / trace(A)` |

`x_hat = pS_in @ data` are the initial internal moments.  `N` is the sensor
noise covariance in physical units and is **not** normalized; only `A` is
normalized in the `mne` variant.  The affine term
`b = mean(x_hat) - B @ S_in @ mean(x_hat)` is applied to both variants, as in
the MNE-Python implementation.

The basis is computed with the MNE-Python defaults used by the integration
(`origin="auto"`, `regularize="in"`, `int_order=8`, `ext_order=3`) and is
shared by both variants.  This differs from the hard-coded basis in the
original script (`origin=(0, 0, 0)`, `regularize=None`); using one basis is
what isolates the effect of the prior.

## Usage

Run both variants and the least-squares SSS reference:

```bash
python compare_variants.py --raw raw.fif --out results
```

Run each variant separately:

```bash
python compare_variants.py --raw raw.fif --method xan --out results/xan
python compare_variants.py --raw raw.fif --method mne --out results/mne
```

Use a separate recording for `N` (for example an empty-room recording), a
saved covariance, or a baseline window:

```bash
python compare_variants.py --raw raw.fif --noise-raw empty.fif \
    --noise-tmin 0 --noise-tmax 60 --out results
python compare_variants.py --raw raw.fif --noise-cov empty-cov.fif --out results
```

Options:

- `--noise-type E|OTP`: empirical covariance or oversampled temporal
  projection (ignored with `--noise-cov`).
- `--noise-cov`: saved covariance used as `N`; channels are reordered to match
  the raw file.
- `--regularize in|in_argmax|None`: shared SSS regularization.  `None` is
  rejected when the basis is too ill-conditioned; use `in` for those files.
- `--origin auto|x,y,z`: shared SSS origin.
- `--tmin/--tmax`: segment used for the comparison.
- `--noise-tmin/--noise-tmax`: segment used for `N` (not with `--noise-cov`).
- `--l-freq/--h-freq`: optional filtering before the comparison.
- `--clean`: optional clean recording for a fidelity reference.
- `--save-fif`: save one FIF per method.

Each run writes `metrics.csv` and `psd.png` to the output directory.

## Example on the testing fixture

`test_move_anon_raw.fif`, 0-4 s, `N` from the same 0-4 s segment, no injected
noise:

| method | LF gain (0.5-3 Hz) | r vs SSS |
|---|---:|---:|
| `sss` | 2.84 | 1.000 |
| `xan` | 42.56 | 0.017 |
| `mne` | 59.61 | 0.388 |

The raw prior (`xan`) over-suppresses when `N` is estimated from the same
clean data.  The de-biased and trace-normalized prior (`mne`) suppresses less
and keeps a higher correlation with standard SSS.

As a fidelity check, injecting the same OPM-like LF artifact as the original
PoC gives:

| method | LF gain | r vs standard SSS |
|---|---:|---:|
| `sss` | 0.12 | 0.942 |
| `xan` | 133.49 | 0.016 |
| `mne` | 96.43 | 0.388 |

The original notebook (`run_foster`, without the affine term) reports
92.9 / 0.396 for the same injected artifact.  The small difference comes from
the affine term and the MNE-Python version; the prior and `N` handling are the
same.

## Scope and limitations

- The comparison isolates the moment prior.  It estimates the data covariance
  with `np.cov` over the selected segment; it does not yet chunk the data,
  exclude annotated segments, or apply cross-talk correction.  The MNE-Python
  integration does those steps separately.
- `fosters_inverse_sss.py` is not modified.  Its `__main__` example currently
  calls `fosters_inverse(raw, Ntmin, Ntmax)` without `ntype`; this branch does
  not use that entry point.
- The original repository has no license file.  These files are provided for
  comparison inside this repository; do not copy the original code elsewhere
  without permission from its author.
