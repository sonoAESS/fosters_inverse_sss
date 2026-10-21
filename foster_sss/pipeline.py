"""End-to-end comparison pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mne

from .basis import compute_basis
from .data import align_channels, load_raw, save_raw
from .metrics import lf_band_power, pearson, plot_psd
from .methods import foster_operator, standard_sss
from .noise import compute_noise_cov


@dataclass
class ComparisonConfig:
    """Configuration for one comparison run."""

    raw: str
    out: str = "results"
    method: str = "both"
    normalize_n: bool = False
    noise_raw: str | None = None
    noise_cov: str | None = None
    clean: str | None = None
    tmin: float | None = None
    tmax: float | None = None
    noise_tmin: float | None = None
    noise_tmax: float | None = None
    l_freq: float | None = None
    h_freq: float | None = None
    noise_type: str = "E"
    regularize: str | None = "in"
    origin: str | tuple[float, float, float] = "auto"
    int_order: int = 8
    ext_order: int = 3
    duration: float = 10.0
    save_fif: bool = False


def selected_priors(method):
    """Return the priors selected by ``--method``."""
    if method == "both":
        return ("foster", "noise_weighted")
    if method in ("foster", "noise_weighted"):
        return (method,)
    return ()


def _method_label(prior, normalize_n):
    return f"{prior}_norm_n" if normalize_n else prior


def _load_noise(config, raw):
    if config.noise_cov is not None:
        cov = mne.read_cov(config.noise_cov)
        cov = mne.pick_channels_cov(
            cov, raw.ch_names, exclude=[], ordered=True, copy=True
        )
        N = np.asarray(cov["data"])
        if N.shape != (len(raw.ch_names), len(raw.ch_names)):
            raise RuntimeError(
                f"noise covariance has shape {N.shape}, expected "
                f"({len(raw.ch_names)}, {len(raw.ch_names)})"
            )
        return N

    if config.noise_raw is not None:
        noise_raw = align_channels(
            load_raw(config.noise_raw, config.l_freq, config.h_freq), raw
        )
    else:
        noise_raw = raw
    return compute_noise_cov(
        noise_raw,
        noise_type=config.noise_type,
        tmin=config.noise_tmin,
        tmax=config.noise_tmax,
        duration=config.duration,
    )


def _evaluate_method(S_in, pS_in, N, phi, *, prior, normalize_n):
    try:
        recon, _, _ = foster_operator(
            S_in,
            pS_in,
            N,
            phi,
            prior=prior,
            affine=True,
            normalize_n=normalize_n,
        )
    except ValueError as exc:
        if "degenerate" not in str(exc):
            raise
        return None, "degenerate", str(exc)
    return recon, "ok", ""


def run_comparison(config):
    """Run standard SSS and the selected Foster variants.

    Parameters
    ----------
    config : ComparisonConfig
        Run configuration.

    Returns
    -------
    result : dict
        Keys ``results`` (method name to reconstructed data), ``rows``
        (metric dictionaries), and ``out_dir``.
    """
    if config.method not in ("both", "foster", "noise_weighted", "sss"):
        raise ValueError(f"Unknown method {config.method!r}")
    if config.noise_cov is not None and (
        config.noise_tmin is not None or config.noise_tmax is not None
    ):
        raise ValueError("noise_tmin/noise_tmax cannot be used with noise_cov")

    raw = load_raw(config.raw, config.l_freq, config.h_freq)
    N = _load_noise(config, raw)
    if config.tmin is not None or config.tmax is not None:
        raw.crop(tmin=config.tmin, tmax=config.tmax)

    try:
        S, pS, reg_moments, n_use_in, coord_frame = compute_basis(
            raw,
            origin=config.origin,
            int_order=config.int_order,
            ext_order=config.ext_order,
            regularize=config.regularize,
        )
    except RuntimeError as exc:
        if config.regularize is None:
            raise RuntimeError(
                "The SSS basis with regularize=None is ill-conditioned for "
                "this file; use regularize='in' (the MNE-Python default)."
            ) from exc
        raise

    S_in = S[:, :n_use_in]
    pS_in = pS[:n_use_in]
    phi = raw.get_data()
    if phi.shape[0] != S_in.shape[0]:
        raise RuntimeError(
            f"Data has {phi.shape[0]} channels but the basis has "
            f"{S_in.shape[0]} rows"
        )

    results = {"sss": standard_sss(S_in, pS_in, phi)}
    statuses = {"sss": ("ok", "")}
    for prior in selected_priors(config.method):
        for normalize_n in (False, True) if config.normalize_n else (False,):
            name = _method_label(prior, normalize_n)
            recon, status, note = _evaluate_method(
                S_in,
                pS_in,
                N,
                phi,
                prior=prior,
                normalize_n=normalize_n,
            )
            statuses[name] = (status, note)
            if recon is not None:
                results[name] = recon

    clean_data = None
    if config.clean is not None:
        clean_raw = align_channels(
            load_raw(config.clean, config.l_freq, config.h_freq), raw
        )
        if config.tmin is not None or config.tmax is not None:
            clean_raw.crop(tmin=config.tmin, tmax=config.tmax)
        clean_data = clean_raw.get_data()
        if clean_data.shape != phi.shape:
            raise RuntimeError(
                f"Clean data shape {clean_data.shape} does not match "
                f"data shape {phi.shape}"
            )
        clean_data = standard_sss(S_in, pS_in, clean_data)

    raw_lf = lf_band_power(phi, raw.info["sfreq"])
    rows = []
    for name, (status, note) in statuses.items():
        if name not in results:
            rows.append(
                {
                    "method": name,
                    "lf_gain": "",
                    "lf_power": "",
                    "r_vs_sss": "",
                    "r_vs_clean": "",
                    "status": status,
                    "note": note,
                }
            )
            continue
        data = results[name]
        lf_power = lf_band_power(data, raw.info["sfreq"])
        row = {
            "method": name,
            "lf_power": lf_power,
            "lf_gain": raw_lf / lf_power,
            "r_vs_sss": np.nanmean(pearson(data, results["sss"])),
            "r_vs_clean": "",
            "status": status,
            "note": note,
        }
        if clean_data is not None:
            row["r_vs_clean"] = np.nanmean(pearson(data, clean_data))
        rows.append(row)

    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "lf_gain",
        "lf_power",
        "r_vs_sss",
        "r_vs_clean",
        "status",
        "note",
    ]
    with (out / "metrics.csv").open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"basis: S={S.shape}, n_use_in={n_use_in}, "
        f"coord_frame={coord_frame}, N={N.shape}"
    )
    header = f"{'method':>23} | {'LF gain':>8} | {'r vs SSS':>9} | status"
    if clean_data is not None:
        header = header.replace(" | status", f" | {'r vs clean':>10} | status")
    print(header)
    for row in rows:
        if row["status"] != "ok":
            line = f"{row['method']:>23} | {'n/a':>8} | {'n/a':>9}"
            if clean_data is not None:
                line += f" | {'n/a':>10}"
            line += f" | {row['status']}"
        else:
            line = (
                f"{row['method']:>23} | {row['lf_gain']:8.2f} | "
                f"{row['r_vs_sss']:+9.3f}"
            )
            if clean_data is not None:
                line += f" | {row['r_vs_clean']:+10.3f}"
            line += f" | {row['status']}"
        print(line)

    plot_psd(results, raw.info["sfreq"], out / "psd.png")
    if config.save_fif:
        for name, data in results.items():
            save_raw(data, raw.info, out / f"{name}_meg.fif")
    print(f"wrote {out / 'metrics.csv'} and {out / 'psd.png'}")
    return {"results": results, "rows": rows, "out_dir": out}
