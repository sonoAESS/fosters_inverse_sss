import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import pinv

from foster_variants import foster_operator, foster_prior, standard_sss


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n_ch, n_mom, n_times = 12, 5, 200
    S = rng.normal(size=(n_ch, n_mom))
    pS = np.linalg.pinv(S)
    data = rng.normal(size=(n_ch, n_times))
    noise = rng.normal(size=(n_ch, n_ch))
    N = noise @ noise.T / n_ch + 0.1 * np.eye(n_ch)
    return S, pS, N, data


def test_xan_prior_is_raw_covariance(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="xan")
    assert_allclose(A, np.cov(x_hat))


def test_mne_prior_is_trace_normalized_and_psd(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="mne", normalize=True, psd=True)
    assert_allclose(np.trace(A), 1.0)
    assert_allclose(A, A.T)
    eigvals = np.linalg.eigvalsh(A)
    assert eigvals.min() >= -1e-12


def test_mne_prior_without_normalization(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="mne", normalize=False, psd=True)
    assert not np.isclose(np.trace(A), 1.0)


def test_affine_term_preserves_mean(synthetic):
    S, pS, N, data = synthetic
    _, B, b = foster_operator(S, pS, N, data, prior="mne", affine=True)
    mean_moments = (pS @ data).mean(axis=1)
    assert_allclose(B @ S @ mean_moments + b, mean_moments)


def test_no_affine_gives_zero_offset(synthetic):
    S, pS, N, data = synthetic
    _, _, b = foster_operator(S, pS, N, data, prior="mne", affine=False)
    assert_allclose(b, 0.0)


def test_noise_covariance_scale_matters(synthetic):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(S, pS, N, data, prior="mne")
    _, B2, _ = foster_operator(S, pS, 2.0 * N, data, prior="mne")
    assert not np.allclose(B1, B2)


def test_shapes_and_finiteness(synthetic):
    S, pS, N, data = synthetic
    recon, B, b = foster_operator(S, pS, N, data, prior="mne")
    assert recon.shape == data.shape
    assert B.shape == (S.shape[1], S.shape[0])
    assert b.shape == (S.shape[1],)
    assert np.isfinite(recon).all()
    assert np.isfinite(B).all()
    assert np.isfinite(b).all()


def test_xan_pseudoinverse_matches_formula(synthetic):
    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="xan")
    A = np.cov(pS @ data)
    B_ref = A @ S.T @ pinv(S @ A @ S.T + N, check_finite=False)
    assert_allclose(B, B_ref)


def test_mne_matches_mne_python(synthetic):
    from mne.preprocessing.maxwell import _foster_inverse

    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="mne")
    moments_cov = pS @ np.cov(data) @ pS.T
    B_ref = _foster_inverse(S, N, moments_cov, pS)
    assert_allclose(B, B_ref, rtol=1e-6, atol=1e-8)


def test_standard_sss(synthetic):
    S, pS, N, data = synthetic
    assert_allclose(standard_sss(S, pS, data), S @ pS @ data)


def test_degenerate_prior_raises(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="degenerate"):
        foster_prior(pS @ data, pS, 1e6 * N, prior="mne")


def test_bad_shapes_raise(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="pS_in"):
        foster_operator(S, pS[:, :-1], N, data)
    with pytest.raises(ValueError, match="N must"):
        foster_operator(S, pS, N[:-1, :-1], data)
    with pytest.raises(ValueError, match="data must"):
        foster_operator(S, pS, N, data[:-1])
    with pytest.raises(ValueError, match="At least two"):
        foster_operator(S, pS, N, data[:, :1])
