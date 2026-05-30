"""
Abstract model interfaces for the mbc toolbox.

Notation follows the ControlToolbox conventions:

* Drift            ``f``
* Diffusion        ``sigma``
* Algebraic        ``g`` (SDAE constraint, ``g(x, y, ...) = 0``)
* Measurement      ``hm`` (discrete noisy measurement function ``h^m``)
* Output           ``gm`` (continuous noiseless output ``g^m``, used in EMPC)

Discrete-time interfaces
------------------------
``DiscreteLinearSDE`` — abstract base for linear discrete-time systems:

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
    ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)

``DiscreteLinearisedSDE`` — extends ``DiscreteLinearSDE`` with a steady-state
operating point and deviation-variable formulation:

    δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k] + Gd w[k],   δx = x − x_s

Continuous-discrete SDE interfaces
------------------------------------
``ContinuousDiscreteSDE`` — abstract base for continuous-discrete
stochastic systems (ControlToolbox §SDE):

    dx(t)   = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    z(t)    = gm(x, u, d, p, t)
    ym(tk)  = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)

``ContinuousDiscreteLinearSDE`` — extends ``ContinuousDiscreteSDE`` for
linear systems where the drift, diffusion, output, and observation functions
take the specific forms:

    f(x, u, d, p, t)     = A x + B u + E d
    sigma(x, u, d, p, t) = G                         (constant diffusion)
    gm(x, u, d, p, t)    = Cz x + Dz u + Fz d
    hm(x, u, d, p, t)    = Cm x + Dm u + Fm d

``ContinuousDiscreteLinearisedSDE`` — extends ``ContinuousDiscreteLinearSDE``
with a steady-state operating point and deviation-variable formulation:

    dδx(t) = (A δx + B δu + E δd) dt + G dw(t),   δx = x − x_s

``ContinuousDiscreteSDAE`` — standalone abstract base for continuous-discrete
stochastic differential-algebraic systems (ControlToolbox §SDAE):

    dx(t)   = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    0       = g(x, y, u, d, p, t)
    z(t)    = gm(x, y, u, d, p, t)
    ym(tk)  = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)
"""

from .discrete_linear_sde import DiscreteLinearSDE
from .discrete_linearised_sde import DiscreteLinearisedSDE
from .continuous_discrete_sde import ContinuousDiscreteSDE
from .continuous_discrete_linear_sde import ContinuousDiscreteLinearSDE
from .continuous_discrete_linearised_sde import ContinuousDiscreteLinearisedSDE
from .continuous_discrete_sdae import ContinuousDiscreteSDAE

__all__ = [
    "DiscreteLinearSDE",
    "DiscreteLinearisedSDE",
    "ContinuousDiscreteSDE",
    "ContinuousDiscreteLinearSDE",
    "ContinuousDiscreteLinearisedSDE",
    "ContinuousDiscreteSDAE",
]
