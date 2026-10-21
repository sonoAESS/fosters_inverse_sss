#!/usr/bin/env python3
"""Foster's inverse variants used to compare against the original script.

The original ``fosters_inverse_sss.py`` is left untouched.  This module adds
the de-biased and trace-normalized prior developed in the MNE-Python
integration, plus a literal port of the original raw-covariance prior so both
can be evaluated with the same SSS basis and the same noise covariance.

``N`` is the sensor noise covariance in physical units and is *not*
normalized.  Only the moment prior ``A`` is normalized (to unit trace) in the
``"mne"`` variant.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, pinv

import mne


def _validate_inputs(S_in, pS_in, N, data):
    S_in = np.asarray(S_in, dtype=float)
    pS_in = np.asarray(pS_in, dtype=float)
    N = np.asarray(N, dtype=float)
    data = np.asarray(data, dtype=float)
    if S_in.ndim != 2:
        raise ValueError(f"S_in must be 2D, got shape {S_in.shape}")
    if pS_in.shape != (S_in.shape[1], S_in.shape[0]):
        raise ValueError(
            f"pS_in must have shape {(S_in.shape[1], S_in.shape[0])}, "
            f"got {pS_in.shape}"
        )
    if N.shape != (S_in.shape[0], S_in.shape[0]):
        raise ValueError(
            f"N must have shape {(S_in.shape[0], S_in.shape[0])}, got {N.shape}"
        )
    if data.ndim != 2 or data.shape[0] != S_in.shape[0]:
        raise ValueError(
            f"data must have shape {(S_in.shape[0], 'n_times')}, got {data.shape}"
        )
    if data.shape[1] < 2:
        raise ValueError("At least two time samples are needed to estimate a covariance")
    return S_in, pS_in, N, data


def foster_prior(x_hat, pS_in, N, *, prior="mne", normalize=True, psd=True):
    """Estimate the moment prior ``A``.

    Parameters
    ----------
    x_hat : ndarray, shape (n_mom, n_times)
        Initial (least-squares) internal moments.
    pS_in : ndarray, shape (n_mom, n_ch)
        Least-squares pseudoinverse of the internal basis.
    N : ndarray, shape (n_ch, n_ch)
        Sensor noise covariance in physical units.  It is used as-is.
    prior : "mne" | "xan"
        ``"mne"`` removes the noise contribution from the data covariance,
        projects onto the PSD cone, and optionally normalizes the trace.
        ``"xan"`` uses the raw covariance, as in the original script.
    normalize : bool
        Normalize ``A`` to unit trace (``"mne"`` only).
    psd : bool
        Project the de-biased covariance onto the PSD cone (``"mne"`` only).

    Returns
    -------
    A : ndarray, shape (n_mom, n_mom)
        Moment prior.
    """
    if prior not in ("mne", "xan"):
        raise ValueError(f"prior must be 'mne' or 'xan', got {prior!r}")
    moments_cov = np.cov(x_hat)
    if prior == "xan":
        return moments_cov
    X = (moments_cov + moments_cov.T) / 2 - pS_in @ N @ pS_in.T
    if psd:
        eigenvalues, eigenvectors = eigh((X + X.T) / 2)
        X = (eigenvectors * np.clip(eigenvalues, 0, None)) @ eigenvectors.T
    trace = np.trace(X)
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError(
            "The Foster's inverse moment prior is degenerate (trace <= 0); "
            "the noise covariance is too large relative to the data covariance."
        )
    return X / trace if normalize else X


def foster_operator(
    S_in,
    pS_in,
    N,
    data,
    *,
    prior="mne",
    normalize=True,
    psd=True,
    affine=True,
):
    """Apply Foster's inverse to the internal subspace.

    Parameters
    ----------
    S_in : ndarray, shape (n_ch, n_mom)
        Internal SSS basis.
    pS_in : ndarray, shape (n_mom, n_ch)
        Least-squares pseudoinverse of ``S_in``.
    N : ndarray, shape (n_ch, n_ch)
        Sensor noise covariance in physical units.  It is not normalized.
    data : ndarray, shape (n_ch, n_times)
        Sensor data of the segment.
    prior : "mne" | "xan"
        Prior variant.  ``"mne"`` is the de-biased, PSD-projected and
        trace-normalized prior; ``"xan"`` is the original raw covariance.
    normalize : bool
        Normalize the ``"mne"`` prior to unit trace.
    psd : bool
        Project the ``"mne"`` prior onto the PSD cone.
    affine : bool
        Add the affine term that preserves the mean moments.

    Returns
    -------
    recon : ndarray, shape (n_ch, n_times)
        Internal reconstruction.
    B : ndarray, shape (n_mom, n_ch)
        Noise-weighted pseudoinverse.
    b : ndarray, shape (n_mom,)
        Affine term (zeros when ``affine=False``).
    """
    S_in, pS_in, N, data = _validate_inputs(S_in, pS_in, N, data)
    x_hat = pS_in @ data
    A = foster_prior(
        x_hat, pS_in, N, prior=prior, normalize=normalize, psd=psd
    )
    B = A @ S_in.T @ pinv(S_in @ A @ S_in.T + N, check_finite=False)
    if affine:
        mean_moments = x_hat.mean(axis=1)
        b = mean_moments - B @ S_in @ mean_moments
    else:
        b = np.zeros(S_in.shape[1])
    moments = B @ data + b[:, None]
    recon = np.real(S_in @ moments)
    return recon, B, b


def standard_sss(S_in, pS_in, data):
    """Least-squares internal SSS reconstruction (reference method)."""
    S_in = np.asarray(S_in, dtype=float)
    pS_in = np.asarray(pS_in, dtype=float)
    data = np.asarray(data, dtype=float)
    if S_in.ndim != 2:
        raise ValueError(f"S_in must be 2D, got shape {S_in.shape}")
    if pS_in.shape != (S_in.shape[1], S_in.shape[0]):
        raise ValueError(
            f"pS_in must have shape {(S_in.shape[1], S_in.shape[0])}, "
            f"got {pS_in.shape}"
        )
    if data.ndim != 2 or data.shape[0] != S_in.shape[0]:
        raise ValueError(
            f"data must have shape {(S_in.shape[0], 'n_times')}, got {data.shape}"
        )
    return np.real(S_in @ (pS_in @ data))


def compute_basis(
    raw,
    *,
    origin="auto",
    int_order=8,
    ext_order=3,
    regularize="in",
    ignore_ref=True,
    mag_scale=100.0,
):
    """Compute the SSS basis using the same parameters for both variants."""
    if raw.info["bads"]:
        raise ValueError(
            "Drop bad channels before computing the basis; got "
            f"{raw.info['bads']}"
        )
    coord_frame = "head" if raw.info["dev_head_t"] is not None else "meg"
    S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(
        raw.info,
        origin=origin,
        int_order=int_order,
        ext_order=ext_order,
        calibration=None,
        coord_frame=coord_frame,
        regularize=regularize,
        ignore_ref=ignore_ref,
        bad_condition="error",
        mag_scale=mag_scale,
        extended_proj=(),
        verbose=False,
    )
    return S, pS, reg_moments, n_use_in, coord_frame


def compute_noise_cov(
    raw,
    *,
    noise_type="E",
    tmin=None,
    tmax=None,
    duration=10.0,
):
    """Compute the sensor noise covariance used by both variants.

    ``noise_type="E"`` follows the original empirical covariance; ``"OTP"``
    follows the oversampled temporal projection estimate.  The returned matrix
    is in physical units and is not normalized.
    """
    if noise_type not in ("E", "OTP"):
        raise ValueError(f"noise_type must be 'E' or 'OTP', got {noise_type!r}")
    raw_meg = raw.copy().pick_types(
        meg=True, eeg=False, eog=False, stim=False, exclude="bads"
    )
    if raw_meg.info["bads"]:
        raise ValueError(
            "Drop bad channels before computing the noise covariance; got "
            f"{raw_meg.info['bads']}"
        )
    if noise_type == "E":
        cov = mne.compute_raw_covariance(
            raw_meg,
            tmin=tmin,
            tmax=tmax,
            rank="info",
            method="empirical",
            verbose=False,
        )
        return np.asarray(cov["data"])
    import sklearn.covariance

    raw_otp = mne.preprocessing.oversampled_temporal_projection(
        raw_meg, duration=duration, verbose=False
    )
    diff = raw_meg.get_data(picks="meg") - raw_otp.get_data(picks="meg")
    return sklearn.covariance.empirical_covariance(diff.T)
