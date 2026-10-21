"""Operations on TCSPC decay histograms."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, deconvolve, fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import pandas as pd
import json


def shift_decay(arr: np.ndarray, n: int) -> np.ndarray:
    """Shift a decay histogram by ``n`` bins, wrapping at the boundaries."""
    return np.roll(arr, -n)



def df_from_parameters(param):
    df = pd.DataFrame(
                {k: param[k] for k in param.keys()}, index =[0]
                )
    return df


from scipy.signal import fftconvolve
from scipy.optimize import minimize

def fit_decay_reconvolution_folded(
        time_axis,
        decay,
        irf,
        fit_limits_channels,
        initial_guess: LifetimeParameters,
        n_exp: int | None = None,
        n_periods=5,
        fixed=None
        ):
    """Fit with folded reconvolution. ``time_axis`` and lifetimes are in seconds."""

    initial_guess = _validate_lifetime_parameters(initial_guess, n_exp)
    n_exp = initial_guess.n_exponentials
    initial_guess = _rescale_parameters(initial_guess, _FIT_TIME_SCALE)
    time_axis = np.asarray(time_axis, dtype=float) * _FIT_TIME_SCALE



    def _pack_tcspc(A, tau, B, shift):
        A = np.asarray(A)
        tau = np.asarray(tau)

        if np.any(A <= 0):
            raise ValueError("A must be positive")
        if np.any(tau <= 0):
            raise ValueError("tau must be positive")
        if B <= 0:
            raise ValueError("B must be positive")

        # ---- total intensity ----
        I = np.sum(A)
        a = np.log(I)

        # ---- weights ----
        w = A / I
        u = np.log(w)

        # ---- rates ----
        lam = 1.0 / tau
        r = np.log(lam)

        # ---- background ----
        b = np.log(B)

        # ---- shift ----
        s = shift

        return np.concatenate([[a], u, r, [b, s]])


    def _unpack_tcspc(x, n_exp):
        x = np.asarray(x)
        idx = 0

        # ---- total intensity ----
        a = x[idx]
        idx += 1
        I = np.exp(a)

        # ---- mixture weights ----
        u = x[idx:idx+n_exp]
        idx += n_exp

        u = u - np.max(u)  # stability
        w = np.exp(u)
        w = w / np.sum(w)

        A = I * w

        # ---- decay rates ----
        r = x[idx:idx+n_exp]
        idx += n_exp

        lam = np.exp(r)
        tau = 1.0 / lam

        # ---- background ----
        b = x[idx]
        idx += 1
        B = np.exp(b)

        # ---- shift ----
        shift = x[idx]

        return A, tau, B, shift

    # ---- helper for NLL from full mu ----
    def _nll_from_mu(mu_full_local):
        mu_loc = mu_full_local[start:end]
        mu_loc = np.clip(mu_loc, 1e-12, None)
        return np.sum(mu_loc - n * np.log(mu_loc))


    # ---- helper to build convolved exponential model ----
    def _forward_model(A,tau, B, shift):
        mu_ext = np.zeros_like(t_ext)
        for j in range(n_exp):
            mu_ext += A[j] * np.exp(-t_ext / tau[j])
        
        irf_shifted = np.interp(t_ext[:len(irf_slice)] - shift, 
                                t_ext[:len(irf_slice)], 
                                irf_slice, 
                                left=0.0, 
                                right=0.0)

        mu_ext = fftconvolve(mu_ext, irf_shifted, mode='full')[:L_ext]
        mu = _fold_periodic(mu_ext, T_bins, n_periods)

        mu += B

        return mu


    def _fold_periodic(signal, T_bins, K):
        """
        Fold a long signal into one period using periodic accumulation.

        Parameters
        ----------
        signal : array
            Length >= (K+1)*T_bins
        T_bins : int
            Period in bins
        K : int
            Number of previous periods to include

        Returns
        -------
        folded : array of length T_bins
        """
        folded = signal[:T_bins].copy()
        for k in range(1, K+1):
            start = k * T_bins
            end = start + T_bins
            folded += signal[start:end]
        return folded

    def _nll_grad(x_free):
        # ---- reconstruct full optimizer vector ----
        x_full = x_full0_optimizer.copy()
        x_full[free_idx] = x_free

        # ---- unpack ----
        A, tau, B, shift = _unpack_tcspc(x_full, n_exp)

        # ---- precompute exponentials ----
        # E = [np.exp(-t / tau[j]) for j in range(n_exp)]
        E = [np.exp(-t_ext / tau[j]) for j in range(n_exp)]

        # ---- extended emission model ----
        m_ext = np.zeros_like(t_ext)
        for j in range(n_exp):
            m_ext += A[j] * E[j]

        # ---- IRF ----
        irf_shifted = np.interp(
            t_ext[:len(irf_slice)] - shift,
            t_ext[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )

        conv_E = []
        for j in range(n_exp):
            conv_ext = fftconvolve(E[j], irf_shifted, mode='full')[:L_ext]
            conv_E.append(_fold_periodic(conv_ext, T_bins, n_periods))

        m_conv = np.zeros_like(t)
        for j in range(n_exp):
            m_conv += A[j] * conv_E[j]

        mu_full = m_conv + B
        mu = mu_full[start:end]
        mu = np.clip(mu, 1e-12, None)

        # f0 = np.sum(mu - n * np.log(mu))

        # ---- gradient container ----
        grad = np.zeros_like(x_full)

        # =========================================================
        # (1) Amplitudes (no FFT)
        # =========================================================
        for j in range(n_exp):
            h = 1e-6 * max(1.0, A[j])

            mu_plus  = mu_full + h * conv_E[j]
            mu_minus = mu_full - h * conv_E[j]

            f_plus  = _nll_from_mu(mu_plus)
            f_minus = _nll_from_mu(mu_minus)

            dA = (f_plus - f_minus) / (2 * h)

            # map to optimizer space later
            grad_A_j = dA

            # store temporarily in physical space slot
            grad[j] = grad_A_j  # placeholder

        # =========================================================
        # (2) Lifetimes (1 FFT each)
        # =========================================================
        for j in range(n_exp):
            h = 1e-6 * tau[j]

            # plus
            tau_p = tau[j] + h
            E_p = np.exp(-t_ext / tau_p)
            # conv_p = fftconvolve(E_p, irf_shifted, mode='full')[:len(t)]

            m_p = m_ext - A[j] * E[j] + A[j] * E_p
            # mu_p = fftconvolve(m_p, irf_shifted, mode='full')[:len(t)] + B
            conv_ext = fftconvolve(m_p, irf_shifted, mode='full')[:L_ext]
            mu_p = _fold_periodic(conv_ext, T_bins, n_exp) + B

            # minus
            tau_m = tau[j] - h
            E_m = np.exp(-t_ext / tau_m)
            # conv_m = fftconvolve(E_m, irf_shifted, mode='full')[:len(t)]

            m_m = m_ext - A[j] * E[j] + A[j] * E_m
            # mu_m = fftconvolve(m_m, irf_shifted, mode='full')[:len(t)] + B
            conv_ext = fftconvolve(m_m, irf_shifted, mode='full')[:L_ext]
            mu_m = _fold_periodic(conv_ext, T_bins, n_exp) + B

            f_plus  = _nll_from_mu(mu_p)
            f_minus = _nll_from_mu(mu_m)

            dtau = (f_plus - f_minus) / (2 * h)

            grad[n_exp + j] = dtau  # placeholder

        # =========================================================
        # (3) Background (no FFT)
        # =========================================================
        h = 1e-6 * max(1.0, B)

        mu_plus  = mu_full + h
        mu_minus = mu_full - h

        f_plus  = _nll_from_mu(mu_plus)
        f_minus = _nll_from_mu(mu_minus)

        dB = (f_plus - f_minus) / (2 * h)

        grad[-2] = dB  # placeholder

        # =========================================================
        # (4) Shift (2 FFT total)
        # =========================================================
        h = 1e-4  # should be ~ fraction of dt

        # plus
        irf_p = np.interp(
            t[:len(irf_slice)] - (shift + h),
            t[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )
        # mu_p = fftconvolve(m, irf_p, mode='full')[:len(t)] + B

        conv_ext = fftconvolve(m_ext, irf_p, mode='full')[:L_ext]
        mu_p = _fold_periodic(conv_ext, T_bins, n_periods) + B

        # minus
        irf_m = np.interp(
            t[:len(irf_slice)] - (shift - h),
            t[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )
        # mu_m = fftconvolve(m, irf_m, mode='full')[:len(t)] + B
        conv_ext = fftconvolve(m_ext, irf_m, mode='full')[:L_ext]
        mu_m = _fold_periodic(conv_ext, T_bins, n_exp) + B

        f_plus  = _nll_from_mu(mu_p)
        f_minus = _nll_from_mu(mu_m)

        dshift = (f_plus - f_minus) / (2 * h)

        grad[-1] = dshift

        # =========================================================
        # (5) map physical → optimizer space
        # =========================================================

        # extract physical gradients
        dA = grad[:n_exp]
        dtau = grad[n_exp:2*n_exp]
        dB = grad[-2]
        dshift = grad[-1]

        # reconstruct helper quantities
        I = np.sum(A)
        w = A / I

        # total intensity
        dI = np.sum(dA * w)
        da = dI * I

        # weights
        dA_dot_w = np.sum(dA * w)
        du = I * w * (dA - dA_dot_w)

        # rates
        dr = -dtau * tau

        # background
        db = dB * B

        # assemble
        grad_opt = np.concatenate([
            [da],
            du,
            dr,
            [db, dshift]
        ])
        # print(grad_opt[free_idx])
        return grad_opt[free_idx]
    
    def _nll(x_free):
        # Reconstruct full vector in optimizer space
        x_full_optimizer = x_full0_optimizer.copy()
        x_full_optimizer[free_idx] = x_free

        # Unpack full vector
        A,tau,B,t_shift = _unpack_tcspc(x_full_optimizer,n_exp)
        mu = _forward_model(A,tau,B,t_shift)
        return _nll_from_mu(mu)


    if fixed is None:
        fixed = {}
    start, end = fit_limits_channels
    t = time_axis.copy()
    n = decay[start:end].astype(float)

    T_bins = len(t) # Period (window width) in bin number units
    L_ext = (n_periods +1) * T_bins
    dt = t[1] - t[0] # time axis is defined precisely
    t_ext = dt * np.arange(L_ext) # extended time axis



    # ----- Build full log-parameter vector -----
    param_names = [f"A_{j}" for j in range(n_exp)] + \
                  [f"tau_{j}" for j in range(n_exp)] + ["B"] + ["shift"]

    x_full0 = np.concatenate((
        initial_guess.amplitudes,
        initial_guess.taus,
        [initial_guess.background, initial_guess.shift],
    ))

    # ----- Pack full vector ------
    x_full0_optimizer = _pack_tcspc(
        x_full0[:n_exp],
        x_full0[n_exp:2*n_exp],
        x_full0[-2],
        x_full0[-1]
    )


    # ----- Identify free indices -----
    # must add one for total intensity
    free_idx = [i + 1 for i, name in enumerate(param_names) if name not in fixed]
    free_idx = np.concatenate([[0], free_idx]) 
    x0_optimizer = x_full0_optimizer[free_idx]

    # Precompute IRF slice
    irf_slice = irf.astype(float)
    irf_slice /= np.sum(irf_slice)  # normalize IRF



    # ----- Perform minimization -----
    # result = minimize(lambda x: _nll(x),
    #                   x0_optimizer,
    #                   jac=lambda x: _nll_grad(x),
    #                   method='L-BFGS-B',
    #                   options={'maxiter': 10000})

    result = minimize(
        lambda x: _nll(x),
        x0=x0_optimizer,
        method='Nelder-Mead',
        options={'maxiter': 10000}
        )
    # ----- Parse message and status -----

    # ----- Final reconstruction -----
    x_final_optimizer = x_full0_optimizer.copy()
    x_final_optimizer[free_idx] = result.x


    A,tau,B,t_shift = _unpack_tcspc(x_final_optimizer,n_exp)

    
    # Build fitted model
    # mu_fit_ext = np.zeros_like(t_ext)
    # fitted = {}


    # for j in range(n_exp):
    #     fitted[f"A_{j}"] = A[j]
    #     fitted[f"tau_{j}"] = tau[j]
    #     mu_fit_ext += A[j] * np.exp(-t_ext / tau[j])

    # irf_shifted = np.interp(t_ext[:len(irf_slice)] - t_shift, t_ext[:len(irf_slice)], irf_slice, left=0.0, right=0.0)

    # mu_fit_ext = fftconvolve(mu_fit_ext, irf_shifted, mode='full')[:L_ext]
    # mu_fit = _fold_periodic(mu_fit_ext,T_bins, n_periods)

    # fitted["B"] = B
    # fitted["shift"] = t_shift
    
    
    # mu_fit += B

    fitted = _rescale_parameters(
        LifetimeParameters(A, tau, background=float(B), shift=float(t_shift)),
        1 / _FIT_TIME_SCALE,
    )

    mu_fit = _forward_model(A,tau,B,t_shift)
    mu_fit = mu_fit[start:end]

    # Calculate weighted residuals
    valid = n > 0
    weights = np.zeros_like(n)
    # weights = 1 / np.sqrt(n[valid])
    weights[valid] = 1 / np.sqrt(n[valid])
    residuals = (n - mu_fit) * weights

    return {
        "fit_parameters": fitted, 
        "fit": mu_fit, 
        "residuals":residuals,
        "optimizer_result":result,
        "limits":fit_limits_channels,
    }



def fit_decay_reconvolution(
        time_axis,
        decay,
        irf,
        fit_limits_channels,
        initial_guess: LifetimeParameters,
        n_exp: int | None = None,
        fixed=None
        ):
    """Fit with reconvolution. ``time_axis`` and lifetimes are in seconds."""

    initial_guess = _validate_lifetime_parameters(initial_guess, n_exp)
    n_exp = initial_guess.n_exponentials
    initial_guess = _rescale_parameters(initial_guess, _FIT_TIME_SCALE)
    time_axis = np.asarray(time_axis, dtype=float) * _FIT_TIME_SCALE



    def _pack_tcspc(A, tau, B, shift):
        A = np.asarray(A)
        tau = np.asarray(tau)

        if np.any(A <= 0):
            raise ValueError("A must be positive")
        if np.any(tau <= 0):
            raise ValueError("tau must be positive")
        if B <= 0:
            raise ValueError("B must be positive")

        # ---- total intensity ----
        I = np.sum(A)
        a = np.log(I)

        # ---- weights ----
        w = A / I
        u = np.log(w)

        # ---- rates ----
        lam = 1.0 / tau
        r = np.log(lam)

        # ---- background ----
        b = np.log(B)

        # ---- shift ----
        s = shift

        return np.concatenate([[a], u, r, [b, s]])


    def _unpack_tcspc(x, n_exp):
        x = np.asarray(x)
        idx = 0

        # ---- total intensity ----
        a = x[idx]
        idx += 1
        I = np.exp(a)

        # ---- mixture weights ----
        u = x[idx:idx+n_exp]
        idx += n_exp

        u = u - np.max(u)  # stability
        w = np.exp(u)
        w = w / np.sum(w)

        A = I * w

        # ---- decay rates ----
        r = x[idx:idx+n_exp]
        idx += n_exp

        lam = np.exp(r)
        tau = 1.0 / lam

        # ---- background ----
        b = x[idx]
        idx += 1
        B = np.exp(b)

        # ---- shift ----
        shift = x[idx]

        return A, tau, B, shift


    if fixed is None:
        fixed = {}
    start, end = fit_limits_channels
    t = time_axis.copy()
    n = decay[start:end].astype(float)

    # ----- Build full log-parameter vector -----
    param_names = [f"A_{j}" for j in range(n_exp)] + \
                  [f"tau_{j}" for j in range(n_exp)] + ["B"] + ["shift"]

    x_full0 = np.concatenate((
        initial_guess.amplitudes,
        initial_guess.taus,
        [initial_guess.background, initial_guess.shift],
    ))

    # ----- Pack full vector ------
    x_full0_optimizer = _pack_tcspc(
        x_full0[:n_exp],
        x_full0[n_exp:2*n_exp],
        x_full0[-2],
        x_full0[-1]
    )


    # ----- Identify free indices -----
    # must add one for total intensity
    free_idx = [i + 1 for i, name in enumerate(param_names) if name not in fixed]
    free_idx = np.concatenate([[0], free_idx]) 
    x0_optimizer = x_full0_optimizer[free_idx]

    # Precompute IRF slice
    irf_slice = irf.astype(float)
    irf_slice /= np.sum(irf_slice)  # normalize IRF

    def _nll_grad(x_free):
        # ---- reconstruct full optimizer vector ----
        x_full = x_full0_optimizer.copy()
        x_full[free_idx] = x_free

        # ---- unpack ----
        A, tau, B, shift = _unpack_tcspc(x_full, n_exp)

        # ---- precompute exponentials ----
        E = [np.exp(-t / tau[j]) for j in range(n_exp)]

        # ---- emission model ----
        m = np.zeros_like(t)
        for j in range(n_exp):
            m += A[j] * E[j]

        # ---- IRF ----
        irf_shifted = np.interp(
            t[:len(irf_slice)] - shift,
            t[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )

        # ---- convolution ----
        conv_E = [
            fftconvolve(E[j], irf_shifted, mode='full')[:len(t)]
            for j in range(n_exp)
        ]

        m_conv = np.zeros_like(t)
        for j in range(n_exp):
            m_conv += A[j] * conv_E[j]

        mu_full = m_conv + B
        mu = mu_full[start:end]
        mu = np.clip(mu, 1e-12, None)

        f0 = np.sum(mu - n * np.log(mu))

        # ---- gradient container ----
        grad = np.zeros_like(x_full)

        # ---- helper for NLL from full mu ----
        def nll_from_mu(mu_full_local):
            mu_loc = mu_full_local[start:end]
            mu_loc = np.clip(mu_loc, 1e-12, None)
            return np.sum(mu_loc - n * np.log(mu_loc))

        # =========================================================
        # (1) Amplitudes (no FFT)
        # =========================================================
        for j in range(n_exp):
            h = 1e-6 * max(1.0, A[j])

            mu_plus  = mu_full + h * conv_E[j]
            mu_minus = mu_full - h * conv_E[j]

            f_plus  = nll_from_mu(mu_plus)
            f_minus = nll_from_mu(mu_minus)

            dA = (f_plus - f_minus) / (2 * h)

            # map to optimizer space later
            grad_A_j = dA

            # store temporarily in physical space slot
            grad[j] = grad_A_j  # placeholder

        # =========================================================
        # (2) Lifetimes (1 FFT each)
        # =========================================================
        for j in range(n_exp):
            h = 1e-6 * tau[j]

            # plus
            tau_p = tau[j] + h
            E_p = np.exp(-t / tau_p)
            # conv_p = fftconvolve(E_p, irf_shifted, mode='full')[:len(t)]

            m_p = m - A[j] * E[j] + A[j] * E_p
            mu_p = fftconvolve(m_p, irf_shifted, mode='full')[:len(t)] + B

            # minus
            tau_m = tau[j] - h
            E_m = np.exp(-t / tau_m)
            # conv_m = fftconvolve(E_m, irf_shifted, mode='full')[:len(t)]

            m_m = m - A[j] * E[j] + A[j] * E_m
            mu_m = fftconvolve(m_m, irf_shifted, mode='full')[:len(t)] + B

            f_plus  = nll_from_mu(mu_p)
            f_minus = nll_from_mu(mu_m)

            dtau = (f_plus - f_minus) / (2 * h)

            grad[n_exp + j] = dtau  # placeholder

        # =========================================================
        # (3) Background (no FFT)
        # =========================================================
        h = 1e-6 * max(1.0, B)

        mu_plus  = mu_full + h
        mu_minus = mu_full - h

        f_plus  = nll_from_mu(mu_plus)
        f_minus = nll_from_mu(mu_minus)

        dB = (f_plus - f_minus) / (2 * h)

        grad[-2] = dB  # placeholder

        # =========================================================
        # (4) Shift (2 FFT total)
        # =========================================================
        h = 1e-4  # should be ~ fraction of dt

        # plus
        irf_p = np.interp(
            t[:len(irf_slice)] - (shift + h),
            t[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )
        mu_p = fftconvolve(m, irf_p, mode='full')[:len(t)] + B

        # minus
        irf_m = np.interp(
            t[:len(irf_slice)] - (shift - h),
            t[:len(irf_slice)],
            irf_slice,
            left=0.0,
            right=0.0
        )
        mu_m = fftconvolve(m, irf_m, mode='full')[:len(t)] + B

        f_plus  = nll_from_mu(mu_p)
        f_minus = nll_from_mu(mu_m)

        dshift = (f_plus - f_minus) / (2 * h)

        grad[-1] = dshift

        # =========================================================
        # (5) map physical → optimizer space
        # =========================================================

        # extract physical gradients
        dA = grad[:n_exp]
        dtau = grad[n_exp:2*n_exp]
        dB = grad[-2]
        dshift = grad[-1]

        # reconstruct helper quantities
        I = np.sum(A)
        w = A / I

        # total intensity
        dI = np.sum(dA * w)
        da = dI * I

        # weights
        dA_dot_w = np.sum(dA * w)
        du = I * w * (dA - dA_dot_w)

        # rates
        dr = -dtau * tau

        # background
        db = dB * B

        # assemble
        grad_opt = np.concatenate([
            [da],
            du,
            dr,
            [db, dshift]
        ])
        # print(grad_opt[free_idx])
        return grad_opt[free_idx]
    
    def _nll(x_free):
        # Reconstruct full vector in optimizer space
        x_full_optimizer = x_full0_optimizer.copy()
        x_full_optimizer[free_idx] = x_free

        # Unpack full vector
        A,tau,B,t_shift = _unpack_tcspc(x_full_optimizer,n_exp)


        # Build raw exponential model
        mu = np.zeros_like(t)
        for j in range(n_exp):
            mu += A[j] * np.exp(-t / tau[j])

        
        irf_shifted = np.interp(t[:len(irf_slice)] - t_shift, 
                                t[:len(irf_slice)], 
                                irf_slice, 
                                left=0.0, 
                                right=0.0)


        mu = fftconvolve(mu, irf_shifted, mode='full')[:len(t)]
        mu = mu[start:end]
        mu += B

        mu = np.clip(mu, 1e-12, None)

        # Negative log-likelihood
        nll = np.sum(mu - n * np.log(mu))

        return nll



    # ----- Perform minimization -----
    # result = minimize(lambda x: _nll(x),
    #                   x0_optimizer,
    #                   jac=lambda x: _nll_grad(x),
    #                   method='L-BFGS-B',
    #                   options={'maxiter': 10000})

    result = minimize(
        lambda x: _nll(x),
        x0=x0_optimizer,
        method='Nelder-Mead',
        options={'maxiter': 10000}
        )

    
    # ----- Final reconstruction -----
    x_final_optimizer = x_full0_optimizer.copy()
    x_final_optimizer[free_idx] = result.x


    A,tau,B,t_shift = _unpack_tcspc(x_final_optimizer,n_exp)

    
    # Build fitted model
    mu_fit = np.zeros_like(t)
    for j in range(n_exp):
        mu_fit += A[j] * np.exp(-t / tau[j])

    irf_shifted = np.interp(t[:len(irf_slice)] - t_shift, t[:len(irf_slice)], irf_slice, left=0.0, right=0.0)

    mu_fit = fftconvolve(mu_fit, irf_shifted, mode='full')[:len(t)]
    fitted = _rescale_parameters(
        LifetimeParameters(A, tau, background=float(B), shift=float(t_shift)),
        1 / _FIT_TIME_SCALE,
    )
    
    
    mu_fit += B
    mu_fit = mu_fit[start:end]

    # Calculate weighted residuals
    valid = n > 0
    weights = 1 / np.sqrt(n[valid])
    residuals = (n - mu_fit) * weights

    return {
        "fit_parameters": fitted, 
        "fit": mu_fit, 
        "residuals":residuals,
        "optimizer_result":result
    }



def birfi(
        decay: np.ndarray,
        smoothing: float | None = None,
        irf_cutoff: int | None=None
):
    N = decay.astype(float)

    if smoothing != None:
        # Perform anscombe-gaussian smoothing
        # variance-stabilizing
        x_trans = 2 * np.sqrt(N + 3/8)

        # smooth
        x_smooth = gaussian_filter1d(x_trans, sigma=smoothing)

        # inverse transform
        N = (x_smooth / 2)**2 - 3/8


    if irf_cutoff == None:
        dN = np.diff(N,n=1,axis=-1)
        irf_cutoff = np.argmin(dN,axis=-1)
        print(f"Estimated IRF cut-off: {irf_cutoff}")
    
    N_cut = N[irf_cutoff:]

    R,_ = deconvolve(N,N_cut)
    R = np.array(R)

    R[R<0] = 0.0

    return R





# def fit_decay_reconvolution(
#         time_axis,
#         decay,
#         irf,
#         fit_limits_channels,
#         initial_guess,
#         n_exp=1,
#         fixed=None
#         ):

#     if fixed is None:
#         fixed = {}
#     start, end = fit_limits_channels
#     t = time_axis.copy()
#     n = decay[start+1:end+1].astype(float)

#     # ----- Build full log-parameter vector -----
#     param_names = [f"A_{j+1}" for j in range(n_exp)] + \
#                   [f"tau_{j+1}" for j in range(n_exp)] + ["B"] + ["shift"]

#     x_full0 = np.array(
#         [np.sign(initial_guess[name]) * np.log(np.abs(initial_guess[name])) for name in param_names[:n_exp]]
#     )

#     x_full0 = np.append(
#         x_full0,
#         [np.log(initial_guess[name]) for name in param_names[n_exp: 2*n_exp + 1]]
#         )

#     x_full0 = np.append(x_full0,initial_guess['shift'])
#     # ----- Identify free indices -----
#     free_idx = [i for i, name in enumerate(param_names) if name not in fixed]
#     x0 = x_full0[free_idx]

#     # Precompute IRF slice / FFT if IRF given
#     irf_slice = irf.astype(float)
#     irf_slice /= np.sum(irf_slice)  # normalize IRF

#     # Compute time bin
#     dt = np.mean(np.diff(time_axis),axis=0)

#     def _shift_irf(irf, dt, shift):
#         # irf: array
#         # dt: time bin width
#         # shift: continuous delay in same units as dt
#         freqs = np.fft.fftfreq(len(irf), d=dt)
#         phase = np.exp(-2j * np.pi * freqs * shift)
#         return np.real(np.fft.ifft(np.fft.fft(irf) * phase))

#     def _nll_and_grad(x_free):
#         # Reconstruct full vector
#         x_full = x_full0.copy()
#         x_full[free_idx] = x_free

#         for i, name in enumerate(param_names[:n_exp]):
#             if name in fixed:
#                 x_full[i] = np.sign(initial_guess[name]) * np.log(np.abs(initial_guess[name]) + 1e-12)

#         for i, name in enumerate(param_names[n_exp:2*n_exp+1]):
#             if name in fixed:
#                 x_full[i] = np.log(initial_guess[name] + 1e-12)

#         if param_names[-1] in fixed:
#             x_full[-1] = initial_guess['shift']

#         # Build raw exponential model
#         mu = np.zeros_like(t)
#         A_list, tau_list = [], []

#         for j in range(n_exp):
#             A = np.sign(x_full[j]) * np.exp(np.abs(x_full[j]))
#             tau = np.exp(x_full[n_exp + j])
#             A_list.append(A)
#             tau_list.append(tau)
#             mu += A * np.exp(-t / tau)

#         mu = np.clip(mu, 1e-12, None)

#         # t_shift = np.exp(x_full[-1])
#         t_shift = x_full[-1]


#         irf_shifted = _shift_irf(irf_slice,dt=dt,shift=t_shift)

#         mu = fftconvolve(mu, irf_shifted, mode='full')[:len(t)]
#         mu = mu[start:end]
#         B = np.exp(x_full[-2])
#         mu += B

#         # Negative log-likelihood
#         nll = np.sum(mu - n * np.log(mu))

#         # Gradient (chain rule)
#         r = 1.0 - n / mu
#         grad_full = np.zeros_like(x_full)

#         for j in range(n_exp):
#             A = A_list[j]
#             tau = tau_list[j]

#             exp_term = np.exp(-t / tau)
#             exp_conv = fftconvolve(exp_term, irf_shifted, mode='full')[:len(t)]
#             exp_conv = exp_conv[start:end]

#             # dL/dA
#             dL_dA = np.sum(r * exp_conv)

#             # chain rule for log A
#             grad_full[j] = dL_dA * A

#             # d/dtau before convolution
#             d_exp_dtau = (t / tau**2) * np.exp(-t / tau)
#             d_exp_dtau_conv = fftconvolve(d_exp_dtau, irf_shifted, mode='full')[:len(t)]
#             d_exp_dtau_conv = d_exp_dtau_conv[start:end]

#             dL_dtau = np.sum(r * A * d_exp_dtau_conv)

#             # chain rule for log tau
#             grad_full[n_exp + j] = dL_dtau * tau

#         # background
#         dL_dB = np.sum(r)
#         grad_full[-2] = dL_dB * B

#         # shift gradient
#         mu_prime = np.zeros_like(mu)
#         mu_prime[1:-1] = (mu[2:] - mu[:-2]) / (2 * dt)
#         mu_prime[0]  = (mu[1] - mu[0]) / dt
#         mu_prime[-1] = (mu[-1] - mu[-2]) / dt

#         dL_dshift = -np.sum(r * mu_prime)
#         grad_full[-1] = dL_dshift   # or * shift if log-param
#         # # print(f"NLL: {nll}")
#         # print(f"Gradient: {grad_full[free_idx]}")
#         # print(f"Vector: {x_free}")


#         return nll, grad_full[free_idx]

#     # ----- Perform minimization -----
#     result = minimize(lambda x: _nll_and_grad(x)[0],
#                       x0,
#                       jac=lambda x: _nll_and_grad(x)[1],
#                       method='L-BFGS-B',
#                       options={'maxiter': 1000})

#     # result = minimize(
#     #     lambda x: _nll_and_grad(x)[0],
#     #     x0=x0,
#     #     method='Nelder-Mead',
#     #     options={'maxiter': 1000}
#     #     )
    
#     # ----- Final reconstruction -----
#     x_final = x_full0.copy()
#     x_final[free_idx] = result.x
    
#     # Build fitted model
#     mu_fit = np.zeros_like(t)
#     fitted = {}

#     for j in range(n_exp):
#         A = np.sign(x_final[j]) * np.exp(np.abs(x_final[j]))
#         tau = np.exp(x_final[n_exp + j])
#         fitted[f"A_{j+1}"] = A
#         fitted[f"tau_{j+1}"] = tau
#         mu_fit += A * np.exp(-t / tau)

#     # t_shift = np.exp(x_final[-1])
#     t_shift = x_final[-1]

#     irf_shifted = _shift_irf(irf_slice,dt=dt,shift=t_shift)
    
#     mu_fit = fftconvolve(mu_fit, irf_shifted, mode='full')[:len(t)]
#     B = np.exp(x_final[-2])
#     fitted["B"] = B
#     fitted["shift"] = t_shift
    
    
#     mu_fit += B
#     mu_fit = mu_fit[start:end]

#     # Calculate weighted residuals
#     valid = n > 0
#     weights = 1 / np.sqrt(n[valid])
#     residuals = (n - mu_fit) * weights

#     return fitted, mu_fit, residuals

def fit_decay_tail(
        time_axis,
        decay,
        fit_limits_channels,
        initial_guess: LifetimeParameters,
        n_exp: int | None = None,
        offset=None,
        fixed=None
        ):

    """Fit a decay tail. ``time_axis``, ``offset`` and lifetimes are in seconds."""

    initial_guess = _validate_lifetime_parameters(initial_guess, n_exp)
    n_exp = initial_guess.n_exponentials
    original_shift = initial_guess.shift
    initial_guess = _rescale_parameters(initial_guess, _FIT_TIME_SCALE)

    if fixed is None:
        fixed = {}
    
    start, end = fit_limits_channels
    
    t = np.asarray(time_axis, dtype=float)[start:end] * _FIT_TIME_SCALE
    if offset is not None:
        t -= offset * _FIT_TIME_SCALE
    else:
        t -= t[start]
    
    n = decay[start:end].astype(float)

    # ----- Build full log-parameter vector -----
    param_names = [f"A_{j}" for j in range(n_exp)] + \
                  [f"tau_{j}" for j in range(n_exp)] + ["B"]

    initial_values = {
        **{f"A_{j}": value for j, value in enumerate(initial_guess.amplitudes)},
        **{f"tau_{j}": value for j, value in enumerate(initial_guess.taus)},
        "B": initial_guess.background,
    }
    x_full0 = np.log(np.array(
        [initial_values[name] for name in param_names], dtype=float
    ) + 1e-12)

    # ----- Identify free indices -----
    free_idx = [i for i, name in enumerate(param_names) if name not in fixed]
    x0 = x_full0[free_idx]

    def _nll_and_grad(x_free):
        # Reconstruct full vector
        x_full = x_full0.copy()
        x_full[free_idx] = x_free
        for i, name in enumerate(param_names):
            if name in fixed:
                x_full[i] = np.log(initial_values[name] + 1e-12)

        # Build raw exponential model
        mu = np.zeros_like(t)
        A_list, tau_list = [], []

        for j in range(n_exp):
            A = np.exp(x_full[j])
            tau = np.exp(x_full[n_exp + j])
            A_list.append(A)
            tau_list.append(tau)
            mu += A * np.exp(-t / tau)

        B = np.exp(x_full[-1])
        mu += B
        mu = np.clip(mu, 1e-12, None)

        # Negative log-likelihood
        nll = np.sum(mu - n * np.log(mu))

        # Gradient (chain rule)
        r = 1.0 - n / mu
        grad_full = np.zeros_like(x_full)

        for j in range(n_exp):
            A = A_list[j]
            tau = tau_list[j]
            exp_term = np.exp(-t / tau)
            dL_dtau = np.sum(r * A * exp_term * (t / tau**2))
            dL_dA = np.sum(r * exp_term)
            
            grad_full[j] = dL_dA * A
            grad_full[n_exp + j] = dL_dtau * tau

        dL_dB = np.sum(r)
        grad_full[-1] = dL_dB * B

        return nll, grad_full[free_idx]

    # ----- Perform minimization -----
    # result = minimize(lambda x: _nll_and_grad(x)[0],
    #                   x0,
    #                   jac=lambda x: _nll_and_grad(x)[1],
    #                   method='L-BFGS-B',
    #                   options={'maxiter': 1000})


    
    result = minimize(
        lambda x: _nll_and_grad(x)[0],
        x0=x0,
        method='Nelder-Mead',
        options={'maxiter': 10000}
        )


    # ----- Final reconstruction -----
    x_final = x_full0.copy()
    x_final[free_idx] = result.x

    # Build fitted model
    mu_fit = np.zeros_like(t)
    fitted_amplitudes = np.empty(n_exp, dtype=float)
    fitted_taus = np.empty(n_exp, dtype=float)

    for j in range(n_exp):
        A = np.exp(x_final[j])
        tau = np.exp(x_final[n_exp + j])
        fitted_amplitudes[j] = A
        fitted_taus[j] = tau
        mu_fit += A * np.exp(-t / tau)

    B = np.exp(x_final[-1])
    fitted = LifetimeParameters(
        fitted_amplitudes,
        fitted_taus / _FIT_TIME_SCALE,
        background=float(B),
        shift=float(original_shift),
    )
    mu_fit += B

    # Calculate weighted residuals
    valid = n > 0
    weights = np.zeros_like(n)
    weights[valid] = 1 / np.sqrt(n[valid])
    residuals = (n - mu_fit) * weights


    return fitted, mu_fit, residuals, result


from dataclasses import dataclass
import numpy as np


@dataclass
class LifetimeParameters:
    amplitudes: np.ndarray
    taus: np.ndarray
    background: float = 0.0
    shift: float = 0.0

    def __post_init__(self):
        self.amplitudes = np.array(
            self.amplitudes, dtype=float, copy=True
        )
        self.taus = np.array(
            self.taus, dtype=float, copy=True
        )

        if self.amplitudes.ndim != 1 or self.taus.ndim != 1:
            raise ValueError("Amplitudes and taus must be 1D arrays.")

        if len(self.amplitudes) == 0:
            raise ValueError("At least one exponential is required.")

        if len(self.amplitudes) != len(self.taus):
            raise ValueError("Amplitudes and taus must have equal lengths.")

        if not np.all(np.isfinite(self.amplitudes)):
            raise ValueError("Amplitudes must be finite.")

        if not np.all(np.isfinite(self.taus)) or np.any(self.taus <= 0):
            raise ValueError("Lifetimes must be finite and positive.")

        if not np.isfinite(self.background) or not np.isfinite(self.shift):
            raise ValueError("Background and shift must be finite.")

        if np.sum(self.amplitudes * self.taus) <= 0:
            raise ValueError("Total integrated decay must be positive.")

        self.amplitudes.setflags(write=False)
        self.taus.setflags(write=False)

    @property
    def n_exponentials(self):
        return len(self.taus)

    def theoretical_phasor(self, frequency):
        """Return normalized (g, s) at modulation frequency in Hz"""
        if not np.isfinite(frequency) or frequency < 0:
            raise ValueError("Frequency must be finite and non-negative.")

        omega = 2 * np.pi * frequency
        wt = omega * self.taus

        weights = self.amplitudes * self.taus
        denominator = np.sum(weights)

        g = np.sum(weights / (1 + wt**2)) / denominator
        s = np.sum(weights * wt / (1 + wt**2)) / denominator

        return g, s

    def decay(self, time):
        """Evaluate I(t) = background + sum(a_i * exp(-(t-shift)/tau_i))."""
        time = np.asarray(time, dtype=float)
        elapsed = time - self.shift

        exponentials = np.exp(
            -np.maximum(elapsed[..., np.newaxis], 0) / self.taus
        )
        signal = np.sum(self.amplitudes * exponentials, axis=-1)
        signal = np.where(elapsed >= 0, signal, 0.0)

        return self.background + signal

    def amplitude_weighted_average(self):
        """Amplitude-weighted mean lifetime for pre-exponential amplitudes."""
        denominator = np.sum(self.amplitudes)

        if denominator <= 0:
            raise ValueError("Sum of amplitudes must be positive.")

        return np.sum(self.amplitudes * self.taus) / denominator

    def intensity_weighted_average(self):
        """Intensity-weighted mean lifetime for pre-exponential amplitudes."""
        weights = self.amplitudes * self.taus
        denominator = np.sum(weights)

        if denominator <= 0:
            raise ValueError("Total integrated decay must be positive.")

        return np.sum(weights * self.taus) / denominator

    def to_dict(self):
        """Return a JSON-serializable dictionary."""
        return {
            "amplitudes": self.amplitudes.tolist(),
            "taus": self.taus.tolist(),
            "background": float(self.background),
            "shift": float(self.shift),
        }

    def save(self, filename):
        """Save parameters to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    def to_array(self):
        """Return parameters as a flat array: amplitudes, taus, background, shift."""
        return np.concatenate((
            self.amplitudes,
            self.taus,
            [self.background, self.shift]
        ))


    @classmethod
    def from_dict(cls, data):
        """Construct a LifetimeParameters instance from a dictionary."""
        return cls(
            amplitudes=data["amplitudes"],
            taus=data["taus"],
            background=data.get("background", 0.0),
            shift=data.get("shift", 0.0),
        )

    @classmethod
    def load(cls, filename):
        """Load parameters from a JSON file."""
        with open(filename, "r") as file:
            return cls.from_dict(json.load(file))


def _validate_lifetime_parameters(
    initial_guess: LifetimeParameters,
    n_exp: int | None,
) -> LifetimeParameters:
    if not isinstance(initial_guess, LifetimeParameters):
        raise TypeError("initial_guess must be a LifetimeParameters instance")

    if n_exp is not None and n_exp != initial_guess.n_exponentials:
        raise ValueError(
            f"n_exp={n_exp} does not match {initial_guess.n_exponentials} parameters"
        )
    return initial_guess


# Fitters work internally in ns (well-scaled for the optimizers); the public
# API, including LifetimeParameters, is in seconds.
_FIT_TIME_SCALE = 1e9


def _rescale_parameters(params: LifetimeParameters, factor: float) -> LifetimeParameters:
    """Return a copy of ``params`` with taus and shift multiplied by ``factor``."""
    return LifetimeParameters(
        params.amplitudes,
        params.taus * factor,
        background=params.background,
        shift=params.shift * factor,
    )

