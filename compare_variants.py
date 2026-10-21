#!/usr/bin/env python3
"""Compare the original Foster's inverse with the MNE-Python variant.

Both methods use the same SSS basis, the same sensor noise covariance ``N``
(in physical units, not normalized), and the same sensor data.  Only the
moment prior changes:

* ``xan``: raw covariance of the initial moments (original script).
* ``mne``: de-biased, PSD-projected, trace-normalized prior (MNE variant).

The affine term is enabled for both, as in the MNE-Python implementation.
The original ``fosters_inverse_sss.py`` is not imported or modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import mne

from foster_variants import (
    compute_basis,
    compute_noise_cov,
    foster_operator,
    standard_sss,
)

mne.set_log_level("ERROR")


def parse_origin(value):
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"origin must be 'auto' or 'x,y,z', got {value!r}"
        ) from exc


def load_raw(path, l_freq=None, h_freq=None):
    raw = mne.io.read_raw_fif(path, preload=True, allow_maxshield=True)
    raw.pick_types(meg=True, eeg=False, eog=False, stim=False, exclude="bads")
    raw.load_data()
    if raw.info["bads"]:
        raise ValueError(f"Bad channels remain in {path}: {raw.info['bads']}")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose=False)
    return raw


def _n_fft(n_times):
    n_fft = min(2048, n_times)
    n_fft -= n_fft % 2
    if n_fft < 2:
        raise ValueError("At least two time samples are needed for the PSD")
    return n_fft


def lf_band_power(data, sfreq):
    n_fft = _n_fft(data.shape[1])
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=40.0,
        n_fft=n_fft,
        verbose=False,
    )
    band = (freqs >= 0.5) & (freqs <= 3.0)
    if not band.any():
        raise ValueError("The 0.5-3 Hz band is empty; provide longer data")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(np.mean(psd, axis=0)[band], freqs[band])


def pearson(a, b):
    a0 = a - a.mean(axis=-1, keepdims=True)
    b0 = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a0**2).sum(axis=-1) * (b0**2).sum(axis=-1))
    out = np.full(denom.shape, np.nan)
    np.divide((a0 * b0).sum(axis=-1), denom, out=out, where=denom > 0)
    return out


def plot_psd(results, sfreq, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, data in results.items():
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=0.5,
            fmax=40.0,
            n_fft=_n_fft(data.shape[1]),
            verbose=False,
        )
        ax.semilogy(freqs, np.mean(psd, axis=0), label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mean over channels)")
    ax.set_title("Foster's inverse variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def align_channels(raw, reference):
    if raw.ch_names != reference.ch_names:
        missing = sorted(set(reference.ch_names) - set(raw.ch_names))
        if missing:
            raise ValueError(
                "Recordings are missing channels from the reference: "
                + ", ".join(missing)
            )
        raw = raw.copy().pick(reference.ch_names, verbose=False)
    return raw


def save_raw(data, info, path):
    raw = mne.io.RawArray(data, info.copy(), verbose=False)
    raw.save(path, overwrite=True, verbose=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="MEG raw file to compare")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--noise-raw",
        default=None,
        help="Optional raw file used to estimate N (e.g. empty room)",
    )
    noise_group.add_argument(
        "--noise-cov",
        default=None,
        help="Optional saved covariance used as N",
    )
    parser.add_argument(
        "--clean", default=None, help="Optional clean raw file for fidelity"
    )
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--noise-tmin", type=float, default=None)
    parser.add_argument("--noise-tmax", type=float, default=None)
    parser.add_argument("--l-freq", type=float, default=None)
    parser.add_argument("--h-freq", type=float, default=None)
    parser.add_argument(
        "--noise-type", choices=("E", "OTP"), default="E", help="Noise estimate"
    )
    parser.add_argument(
        "--method",
        choices=("both", "xan", "mne", "sss"),
        default="both",
        help="Method(s) to run; 'both' runs xan and mne",
    )
    parser.add_argument(
        "--regularize",
        choices=("in", "in_argmax", "None"),
        default="in",
        help="SSS regularization (same for both variants)",
    )
    parser.add_argument("--origin", type=parse_origin, default="auto")
    parser.add_argument("--int-order", type=int, default=8)
    parser.add_argument("--ext-order", type=int, default=3)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="OTP duration in seconds (only for --noise-type OTP)",
    )
    parser.add_argument(
        "--out", default="comparison_results", help="Output directory"
    )
    parser.add_argument(
        "--save-fif", action="store_true", help="Save one FIF per method"
    )
    args = parser.parse_args()

    regularize = None if args.regularize == "None" else args.regularize
    raw = load_raw(args.raw, args.l_freq, args.h_freq)
    if args.noise_cov is not None:
        if args.noise_tmin is not None or args.noise_tmax is not None:
            parser.error("--noise-tmin/--noise-tmax cannot be used with --noise-cov")
        cov = mne.read_cov(args.noise_cov)
        cov = mne.pick_channels_cov(
            cov, raw.ch_names, exclude=[], ordered=True, copy=True
        )
        N = np.asarray(cov["data"])
        if N.shape != (len(raw.ch_names), len(raw.ch_names)):
            raise RuntimeError(
                f"noise covariance has shape {N.shape}, expected "
                f"({len(raw.ch_names)}, {len(raw.ch_names)})"
            )
    else:
        noise_raw = (
            align_channels(load_raw(args.noise_raw, args.l_freq, args.h_freq), raw)
            if args.noise_raw is not None
            else raw
        )
        N = compute_noise_cov(
            noise_raw,
            noise_type=args.noise_type,
            tmin=args.noise_tmin,
            tmax=args.noise_tmax,
            duration=args.duration,
        )
    if args.tmin is not None or args.tmax is not None:
        raw.crop(tmin=args.tmin, tmax=args.tmax)

    try:
        S, pS, reg_moments, n_use_in, coord_frame = compute_basis(
            raw,
            origin=args.origin,
            int_order=args.int_order,
            ext_order=args.ext_order,
            regularize=regularize,
        )
    except RuntimeError as exc:
        if regularize is None:
            raise RuntimeError(
                "The SSS basis with --regularize None is ill-conditioned for "
                "this file; use --regularize in (the MNE-Python default)."
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
    if args.method in ("both", "xan"):
        results["xan"], _, _ = foster_operator(
            S_in, pS_in, N, phi, prior="xan", affine=True
        )
    if args.method in ("both", "mne"):
        results["mne"], _, _ = foster_operator(
            S_in, pS_in, N, phi, prior="mne", affine=True
        )

    clean_data = None
    if args.clean is not None:
        clean_raw = align_channels(
            load_raw(args.clean, args.l_freq, args.h_freq), raw
        )
        if args.tmin is not None or args.tmax is not None:
            clean_raw.crop(tmin=args.tmin, tmax=args.tmax)
        clean_data = clean_raw.get_data()
        if clean_data.shape != phi.shape:
            raise RuntimeError(
                f"Clean data shape {clean_data.shape} does not match "
                f"data shape {phi.shape}"
            )
        clean_data = standard_sss(S_in, pS_in, clean_data)

    raw_lf = lf_band_power(phi, raw.info["sfreq"])
    rows = []
    for name, data in results.items():
        lf_power = lf_band_power(data, raw.info["sfreq"])
        row = {
            "method": name,
            "lf_power": lf_power,
            "lf_gain": raw_lf / lf_power,
            "r_vs_sss": np.nanmean(pearson(data, results["sss"])),
        }
        if clean_data is not None:
            row["r_vs_clean"] = np.nanmean(pearson(data, clean_data))
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "lf_gain", "lf_power", "r_vs_sss", "r_vs_clean"]
    for row in rows:
        row.setdefault("r_vs_clean", "")
    with (out / "metrics.csv").open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"basis: S={S.shape}, n_use_in={n_use_in}, "
        f"coord_frame={coord_frame}, N={N.shape}"
    )
    header = f"{'method':>8} | {'LF gain':>8} | {'r vs SSS':>9}"
    if clean_data is not None:
        header += f" | {'r vs clean':>10}"
    print(header)
    for row in rows:
        line = (
            f"{row['method']:>8} | {row['lf_gain']:8.2f} | "
            f"{row['r_vs_sss']:+9.3f}"
        )
        if clean_data is not None:
            line += f" | {row['r_vs_clean']:+10.3f}"
        print(line)

    plot_psd(results, raw.info["sfreq"], out / "psd.png")
    if args.save_fif:
        for name, data in results.items():
            save_raw(data, raw.info, out / f"{name}_meg.fif")
    print(f"wrote {out / 'metrics.csv'} and {out / 'psd.png'}")


if __name__ == "__main__":
    main()
