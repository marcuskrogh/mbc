"""
Delayed-Observation Filter wrapper (``DelayedObservationFilter``).

A transparent wrapper that adds per-channel reporting-delay handling to
**any** estimator that implements the unified

    estimator.step(ym, u, d, p, t, mask=None) → (x_hat, P, …)

interface.

Supported wrapped estimators
-----------------------------
* :class:`~mbc.estimation.DiscreteLinearKF`
* :class:`~mbc.estimation.ContinuousDiscreteLinearKF`
* :class:`~mbc.estimation.ContinuousDiscreteEKF`
* :class:`~mbc.estimation.ContinuousDiscreteUKF`
* :class:`~mbc.estimation.ContinuousDiscreteEnKF`
* :class:`~mbc.estimation.ContinuousDiscretePF`

Algorithm (M.Sc. thesis §1.2)
-----------------------------
At each call to :meth:`step` (with measurement ``ym[k]``, input ``u[k-1]``,
disturbance ``d[k-1]``, optional parameters ``p`` and time ``t_k``):

1. **Immediate update** — apply the wrapped estimator with the active
   non-delayed channels (or the full mask when ``delay`` is ``None``).
   Push ``{x_hat, P, ym, u, d, p, t, mask}`` onto a ring buffer
   (``deque`` with ``maxlen = lag_max``).

2. **Delayed corrections** — for each channel ``i`` with ``delay[i] = τ > 0``
   (sorted ascending so shorter lags are processed first):

   a. Restore the estimator state to the posterior at step ``k − τ``.
   b. Apply a measurement-only correction for channel ``i``.
   c. Replay buffered steps ``k − τ + 1 … k`` to rebuild the posterior chain.

3. After all delayed corrections the estimator holds the fully-corrected
   current estimate.

If ``delay[i] > lag_max`` or ``delay[i] >= buffer depth``, channel ``i``
is dropped with a :class:`RuntimeWarning`.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any

import numpy as np

from .._utils import _any_to_np1d, _any_to_np2d, _cholesky_psd


# ── Internal helpers ─────────────────────────────────────────────────────────


def _as_np1d(v) -> np.ndarray | None:
    if v is None:
        return None
    return _any_to_np1d(v).copy()


def _as_np2d(M) -> np.ndarray:
    return _any_to_np2d(M).copy()


def _ny_of(y) -> int:
    return int(_any_to_np1d(y).shape[0])


# ── DelayedObservationFilter ─────────────────────────────────────────────────


class DelayedObservationFilter:
    """
    Transparent per-channel-delay wrapper for any state estimator that
    exposes the unified ``step(ym, u, d, p, t, mask=None)`` API.

    Parameters
    ----------
    estimator : any supported estimator
        Wrapped estimator.
    lag_max : int
        Maximum reporting delay in sampling steps the buffer can
        accommodate.  Channels with a delay exceeding ``lag_max`` (or the
        current buffer depth) are dropped with a :class:`RuntimeWarning`.
    """

    def __init__(self, estimator: Any, lag_max: int) -> None:
        self._est = estimator
        self._lag_max = lag_max
        self._buf: deque[dict] = deque(maxlen=lag_max)

    # ── Properties delegated to the wrapped estimator ───────────────────────

    @property
    def x_hat(self):
        return self._est.x_hat

    @property
    def P(self):
        return self._est.P

    @property
    def last_innovation(self):
        return getattr(self._est, "last_innovation", None)

    # ── State save / restore ──────────────────────────────────────────────

    def _get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return _as_np1d(self._est.x_hat), _as_np2d(self._est.P)

    def _set_state(self, x_np: np.ndarray, P_np: np.ndarray) -> None:
        """
        Restore the wrapped estimator's ``(x_hat, P)`` state.

        For ensemble / particle estimators the particle cloud is resampled
        from N(x_np, P_np); for all other estimators ``_x`` and ``_P``
        are set directly.
        """
        from ._base import ContinuousDiscreteEstimator, DiscreteEstimator
        from .continuous_discrete_enkf import ContinuousDiscreteEnKF
        from .continuous_discrete_pf import ContinuousDiscretePF

        est = self._est
        if isinstance(est, (ContinuousDiscreteEnKF, ContinuousDiscretePF)):
            try:
                nx, N = est._nx, est._N
                rng = est._rng
            except AttributeError as exc:
                raise TypeError(
                    f"DelayedObservationFilter: ensemble/particle estimator "
                    f"{type(est)!r} does not expose expected internal fields "
                    f"(_nx, _N, _rng): {exc}"
                ) from exc
            L = _cholesky_psd(P_np)
            Z = rng.standard_normal((nx, N))
            est._X = x_np[:, None] + L @ Z
        elif isinstance(est, (DiscreteEstimator, ContinuousDiscreteEstimator)):
            est._x = x_np.copy()
            est._P = P_np.copy()
        else:
            raise TypeError(
                f"DelayedObservationFilter: unsupported estimator type {type(est)!r}"
            )

    # ── Channel partitioning ────────────────────────────────────────────────

    @staticmethod
    def _partition(
        ny: int,
        mask,
        delay: np.ndarray | None,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        """
        Split output channels into an *immediate* group and a *delayed* group.

        Returns
        -------
        imm_mask : (ny,) bool ndarray or ``None``
            Active-channel mask for the immediate update.  ``None`` means
            all channels are immediate (``delay`` was ``None``).
        delayed : list of (channel_idx, tau)
            Delayed channels sorted ascending by ``tau``.
        """
        if delay is None:
            if mask is None:
                return None, []
            return np.asarray(mask, dtype=bool), []

        delay_arr = np.asarray(delay, dtype=int)
        user_active = (
            np.ones(ny, dtype=bool)
            if mask is None else np.asarray(mask, dtype=bool)
        )

        imm_mask = np.array(
            [bool(user_active[i]) and int(delay_arr[i]) == 0 for i in range(ny)],
            dtype=bool,
        )
        delayed = sorted(
            [
                (i, int(delay_arr[i]))
                for i in range(ny)
                if user_active[i] and delay_arr[i] > 0
            ],
            key=lambda x: x[1],
        )
        return imm_mask, delayed

    # ── Single-channel measurement-only correction ─────────────────────────

    def _correction_single_channel(
        self,
        ch_idx: int,
        ym_np: np.ndarray,
        x0_np: np.ndarray,
        P0_np: np.ndarray,
        ny: int,
    ) -> None:
        """
        Apply a measurement-only update at the prior ``(x0, P0)`` for
        a single output channel.
        """
        from ._base import ContinuousDiscreteEstimator

        self._set_state(x0_np, P0_np)
        single_mask = np.zeros(ny, dtype=bool)
        single_mask[ch_idx] = True

        if isinstance(self._est, ContinuousDiscreteEstimator):
            entry = self._buf[-1]
            u_e = entry["u"]
            d_e = entry["d"]
            p_e = entry.get("p")
            self._est.update(ym_np, u_e, d_e, p_e, mask=single_mask)
        else:
            self._est.update(ym_np, mask=single_mask)

    # ── Replay forward through buffer ──────────────────────────────────────

    def _replay(self, entries: list[dict]) -> None:
        """Re-propagate the estimator through *entries* to rebuild the chain."""
        for entry in entries:
            self._est.step(
                entry["ym"],
                entry["u"],
                entry["d"],
                entry.get("p"),
                entry.get("t"),
                mask=entry["mask"],
            )
            x_np, P_np = self._get_state()
            entry["x_hat"] = x_np
            entry["P"] = P_np

    # ── Apply delayed corrections ──────────────────────────────────────────

    def _apply_delayed(
        self,
        ym_np: np.ndarray,
        delayed_chs: list[tuple[int, int]],
    ) -> None:
        ny = len(ym_np)
        buf = list(self._buf)
        n_buf = len(buf)

        for ch_idx, tau in delayed_chs:
            if tau >= n_buf:
                warnings.warn(
                    f"DelayedObservationFilter: delay {tau} exceeds buffer "
                    f"depth {n_buf - 1}; dropping channel {ch_idx}.",
                    RuntimeWarning,
                    stacklevel=4,
                )
                continue

            prior = buf[-(tau + 1)]
            self._correction_single_channel(
                ch_idx, ym_np, prior["x_hat"], prior["P"], ny
            )

            x_np, P_np = self._get_state()
            prior["x_hat"] = x_np
            prior["P"] = P_np

            self._replay(buf[-tau:])

        self._buf.clear()
        for e in buf:
            self._buf.append(e)

    # ── Public step interface ──────────────────────────────────────────────

    def step(
        self,
        ym,
        u,
        d,
        p=None,
        t: float | None = None,
        mask=None,
        delay: np.ndarray | None = None,
    ):
        """
        Combined predict + update step with per-channel delay handling.

        Parameters
        ----------
        ym    : (nym,) ndarray — measurement at ``t_k``.
        u     : (nu,) ndarray  — input over the previous interval.
        d     : (nd,) ndarray  — disturbance over that interval.
        p     : (np,) ndarray, optional          — parameter vector.
        t     : float, optional                  — current time.
        mask  : (nym,) bool array, optional      — active-channel mask.
        delay : (nym,) int ndarray, optional     — per-channel delay in steps.
                ``delay[i] = 0`` (or ``None``) means the channel is on time.

        Returns
        -------
        (x_hat, P) — same convention as the wrapped estimator's ``step``.
        """
        ny = _ny_of(ym)
        imm_mask, delayed_chs = self._partition(ny, mask, delay)

        ym_np = _as_np1d(ym)
        u_np  = _as_np1d(u)
        d_np  = _as_np1d(d)
        p_np  = (
            np.asarray(p, dtype=float).ravel() if p is not None else None
        )

        # 1. Immediate step.
        result = self._est.step(ym, u, d, p_np, t, mask=imm_mask)

        x_np, P_np = self._get_state()
        entry: dict = {
            "x_hat": x_np,
            "P": P_np,
            "ym": ym_np,
            "u": u_np,
            "d": d_np,
            "p": p_np,
            "t": t,
            "mask": imm_mask,
        }
        self._buf.append(entry)

        # 2. Delayed corrections.
        if delayed_chs:
            self._apply_delayed(ym_np, delayed_chs)
            x_np, P_np = self._get_state()
            if isinstance(result, tuple):
                result = (x_np.copy(), P_np.copy()) + tuple(result[2:])
            else:
                result = (x_np.copy(), P_np.copy())

        return result
