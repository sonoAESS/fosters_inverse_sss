import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import pinv

from foster_sss.methods import foster_operator, foster_prior, standard_sss
from foster_sss.noise import normalize_noise_cov

PRIORS = ("foster", "noise_weighted")


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


def test_foster_prior_is_raw_covariance(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="foster")
    assert_allclose(A, np.cov(x_hat))


def test_noise_weighted_prior_is_trace_normalized_and_psd(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="noise_weighted", normalize=True, psd=True)
    assert_allclose(np.trace(A), 1.0)
    assert_allclose(A, A.T)
    eigvals = np.linalg.eigvalsh(A)
    assert eigvals.min() >= -1e-12


def test_noise_weighted_prior_without_normalization(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="noise_weighted", normalize=False, psd=True)
    assert not np.isclose(np.trace(A), 1.0)


def test_affine_term_preserves_mean(synthetic):
    S, pS, N, data = synthetic
    _, B, b = foster_operator(S, pS, N, data, prior="noise_weighted", affine=True)
    mean_moments = (pS @ data).mean(axis=1)
    assert_allclose(B @ S @ mean_moments + b, mean_moments)


def test_no_affine_gives_zero_offset(synthetic):
    S, pS, N, data = synthetic
    _, _, b = foster_operator(S, pS, N, data, prior="noise_weighted", affine=False)
    assert_allclose(b, 0.0)


def test_noise_covariance_scale_matters(synthetic):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(S, pS, N, data, prior="noise_weighted")
    _, B2, _ = foster_operator(S, pS, 2.0 * N, data, prior="noise_weighted")
    assert not np.allclose(B1, B2)


def test_shapes_and_finiteness(synthetic):
    S, pS, N, data = synthetic
    recon, B, b = foster_operator(S, pS, N, data, prior="noise_weighted")
    assert recon.shape == data.shape
    assert B.shape == (S.shape[1], S.shape[0])
    assert b.shape == (S.shape[1],)
    assert np.isfinite(recon).all()
    assert np.isfinite(B).all()
    assert np.isfinite(b).all()


def test_foster_pseudoinverse_matches_formula(synthetic):
    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="foster")
    A = np.cov(pS @ data)
    B_ref = A @ S.T @ pinv(S @ A @ S.T + N, check_finite=False)
    assert_allclose(B, B_ref)


def test_noise_weighted_matches_mne_python(synthetic):
    maxwell = pytest.importorskip("mne.preprocessing.maxwell")
    _foster_inverse = getattr(maxwell, "_foster_inverse", None)
    if _foster_inverse is None:
        pytest.skip("MNE-Python does not expose _foster_inverse")

    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="noise_weighted")
    moments_cov = pS @ np.cov(data) @ pS.T
    B_ref = _foster_inverse(S, N, moments_cov, pS)
    assert_allclose(B, B_ref, rtol=1e-6, atol=1e-8)


def test_standard_sss(synthetic):
    S, pS, N, data = synthetic
    assert_allclose(standard_sss(S, pS, data), S @ pS @ data)


def test_degenerate_prior_raises(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="degenerate"):
        foster_prior(pS @ data, pS, 1e6 * N, prior="noise_weighted")


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


def test_normalize_noise_cov_has_unit_trace(synthetic):
    _, _, N, _ = synthetic
    N_norm = normalize_noise_cov(N)
    assert_allclose(np.trace(N_norm), 1.0)
    assert_allclose(N_norm, N_norm.T)


def test_normalize_noise_cov_is_scale_invariant(synthetic):
    _, _, N, _ = synthetic
    assert_allclose(normalize_noise_cov(10.0 * N), normalize_noise_cov(N))


def test_normalize_noise_cov_rejects_nonpositive_trace(synthetic):
    _, _, N, _ = synthetic
    with pytest.raises(ValueError, match="positive trace"):
        normalize_noise_cov(-N)
    with pytest.raises(ValueError, match="square"):
        normalize_noise_cov(N[:-1])


@pytest.mark.parametrize("prior", PRIORS)
def test_normalize_n_matches_normalized_input(synthetic, prior):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(
        S, pS, N, data, prior=prior, normalize_n=True
    )
    _, B2, _ = foster_operator(
        S, pS, normalize_noise_cov(N), data, prior=prior
    )
    assert_allclose(B1, B2)


@pytest.mark.parametrize("prior", PRIORS)
def test_normalize_n_is_scale_invariant(synthetic, prior):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(
        S, pS, N, data, prior=prior, normalize_n=True
    )
    _, B2, _ = foster_operator(
        S, pS, 10.0 * N, data, prior=prior, normalize_n=True
    )
    assert_allclose(B1, B2)


@pytest.mark.parametrize("prior", PRIORS)
def test_normalize_n_changes_result(synthetic, prior):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(S, pS, N, data, prior=prior)
    _, B2, _ = foster_operator(
        S, pS, N, data, prior=prior, normalize_n=True
    )
    assert not np.allclose(B1, B2)


def test_normalize_n_can_degenerate_noise_weighted(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="degenerate"):
        foster_operator(
            S, pS, N, 1e-6 * data, prior="noise_weighted", normalize_n=True
        )
