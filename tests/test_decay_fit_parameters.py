from types import SimpleNamespace

import numpy as np
import pytest

import tttrkit.analysis.decay as decay_module
from tttrkit.analysis.decay import (
    LifetimeParameters,
    fit_decay_reconvolution,
    fit_decay_reconvolution_folded,
    fit_decay_tail,
)


def _stub_minimize(_objective, x0, **_kwargs):
    return SimpleNamespace(x=np.asarray(x0).copy())


@pytest.fixture
def fit_data(monkeypatch):
    monkeypatch.setattr(decay_module, "minimize", _stub_minimize)
    time = np.arange(32, dtype=float) * 0.1
    initial = LifetimeParameters(
        amplitudes=np.array([8.0, 2.0]),
        taus=np.array([0.4, 1.2]),
        background=1.0,
        shift=0.0,
    )
    decay = initial.decay(time)
    irf = np.zeros_like(time)
    irf[0] = 1.0
    return time, decay, irf, initial


def test_folded_reconvolution_accepts_and_returns_lifetime_parameters(fit_data):
    time, decay, irf, initial = fit_data
    result = fit_decay_reconvolution_folded(
        time, decay, irf, (0, len(time)), initial, n_periods=2
    )

    assert isinstance(result["fit_parameters"], LifetimeParameters)
    assert result["fit_parameters"].n_exponentials == 2


def test_reconvolution_accepts_and_returns_lifetime_parameters(fit_data):
    time, decay, irf, initial = fit_data
    result = fit_decay_reconvolution(
        time, decay, irf, (0, len(time)), initial
    )

    assert isinstance(result["fit_parameters"], LifetimeParameters)
    assert result["fit_parameters"].n_exponentials == 2


def test_tail_fit_accepts_and_returns_lifetime_parameters(fit_data):
    time, decay, _irf, initial = fit_data
    fitted, fit, residuals, _optimizer_result = fit_decay_tail(
        time, decay, (0, len(time)), initial
    )

    assert isinstance(fitted, LifetimeParameters)
    assert fitted.n_exponentials == 2
    assert fit.shape == residuals.shape == decay.shape


def test_fits_work_in_seconds_and_preserve_units():
    time = np.arange(256) * 5e-11
    true = LifetimeParameters([500.0, 200.0], [0.5e-9, 2.5e-9], background=2.0)
    decay = true.decay(time)
    guess = LifetimeParameters([400.0, 300.0], [0.7e-9, 2.0e-9], background=1.0)

    fitted, _fit, _res, _opt = fit_decay_tail(
        time, np.random.default_rng(0).poisson(decay).astype(float),
        (0, len(time)), guess, offset=0.0,
    )

    np.testing.assert_allclose(np.sort(fitted.taus), true.taus, rtol=0.15)
    assert fitted.taus.max() < 1e-6
    g, s = fitted.theoretical_phasor(80e6)
    assert 0 < s < 1 and 0 < g < 1


def test_non_lifetime_parameters_guess_is_rejected(fit_data):
    time, decay, _irf, _initial = fit_data
    legacy_guess = {
        "A_0": 8.0,
        "A_1": 2.0,
        "tau_0": 0.4,
        "tau_1": 1.2,
        "B": 1.0,
    }

    with pytest.raises(TypeError, match="LifetimeParameters"):
        fit_decay_tail(time, decay, (0, len(time)), legacy_guess)


def test_component_count_mismatch_is_rejected(fit_data):
    time, decay, irf, initial = fit_data

    with pytest.raises(ValueError, match="does not match"):
        fit_decay_reconvolution(
            time, decay, irf, (0, len(time)), initial, n_exp=1
        )


def test_lifetime_parameters_json_round_trip(tmp_path):
    parameters = LifetimeParameters(
        amplitudes=np.array([3.0, 1.0]),
        taus=np.array([0.5, 1.5]),
        background=2.0,
        shift=0.1,
    )
    path = tmp_path / "lifetime-parameters.json"

    parameters.save(path)
    loaded = LifetimeParameters.load(path)

    np.testing.assert_array_equal(loaded.amplitudes, parameters.amplitudes)
    np.testing.assert_array_equal(loaded.taus, parameters.taus)
    assert loaded.background == parameters.background
    assert loaded.shift == parameters.shift