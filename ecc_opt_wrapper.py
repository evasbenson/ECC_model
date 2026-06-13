"""
ecc_opt_wrapper.py

Translated from ECC_opt_wrapper_v6b.m (latest version from advisor)

Key changes from previous version:
- make_capital uses 10^6 scaling (not 1000)
- K initialized directly from make_capital(M_a0) — no delta_m_avg scaling
- Convergence on percent change in M and K (scale-independent)
- Single fixed tolerance 1e-5 regardless of population
- max_iter = 1000
"""

import numpy as np
import warnings
from datetime import datetime
from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization


def recycle(M_a, M_max, savings, tech, parameters):
    """
    Translated from recycle() in ECC_opt_wrapper_v6b.m

    Computes gross additions to active material stocks from recycling.

    Parameters
    ----------
    M_a : array (5,) — current active material stocks (Gt)
    M_max : array (5,) — maximum accessible material stocks (Gt)
    savings : float — savings = s * Y
    tech : float — technology level tau in [0, 1]
    parameters : dict

    Returns
    -------
    recycled : array (5,) — gross material additions
    """
    rho_max      = parameters['rho_max']
    rho_current  = parameters['rho_current']
    beta_max     = parameters['beta_max']
    beta_current = parameters['beta_current']
    zeta_max     = parameters['zeta_max']
    zeta_current = parameters['zeta_current']
    c3           = parameters['c3']
    delta_m      = parameters['d_m']

    rho  = tech * (rho_max  - rho_current)  + rho_current
    beta = tech * (beta_max - beta_current) + beta_current
    zeta = tech * (zeta_max - zeta_current) + zeta_current

    M_i         = M_max - M_a
    rec_direct  = beta * delta_m * M_a
    rec_diffuse = zeta * M_i
    recycled    = rho * (rec_direct + rec_diffuse) * np.tanh(c3 * savings)

    return recycled


def make_capital(material, parameters):
    """
    Translated from make_capital() in ECC_opt_wrapper_v6b.m

    Converts material vector to a scalar capital stock.
    Uses 10^6 scaling to match MATLAB.

    If phi_subs == 0: log-linear (Cobb-Douglas) aggregation
    Otherwise: CES aggregation

    Parameters
    ----------
    material : array (5,) — material inputs
    parameters : dict

    Returns
    -------
    float : capital value
    """
    phi_share = parameters['phi_share']
    phi_subs  = parameters['phi_subs']

    if phi_subs == 0:
        log_investment = np.dot(phi_share, np.log(material))
        return 1e6 * np.exp(log_investment)
    else:
        return 1e6 * (np.dot(phi_share, material ** phi_subs)) ** (1.0 / phi_subs)


def run_ecc_simulation(tau=None, pop=None, max_iter=1000):
    """
    Runs the full ECC simulation across a technology x population grid.
    Translated from ECC_opt_wrapper_v6b.m (latest version).

    Parameters
    ----------
    tau : array — technology levels (default: [0])
    pop : array — population levels in billions (default: [8])
    max_iter : int — maximum fixed-point iterations (default: 1000)

    Returns
    -------
    utilmat : (len(tau), len(pop)) array — utility at each grid point
    solutioncube : (12, len(tau), len(pop)) array — solutions
    tau, pop : arrays used
    parameters : dict
    """
    parameters = get_parameters()

    s       = parameters['s']
    delta_k = parameters['d_k']
    delta_m = parameters['d_m']
    M_a0    = parameters['M_a0']
    M_max   = parameters['M_max']

    if tau is None:
        tau = np.array([0.0])
    else:
        tau = np.asarray(tau, dtype=float)
    if pop is None:
        pop = np.array([8.0])
    else:
        pop = np.asarray(pop, dtype=float)

    n_tau = len(tau)
    n_pop = len(pop)
    total = n_tau * n_pop
    print(f'grid: {n_tau} tau x {n_pop} pop = {total} problems')

    utilmat      = np.zeros((n_tau, n_pop))
    solutioncube = np.zeros((12, n_tau, n_pop))

    # single fixed tolerance — scale-independent percent change
    tolerance = 1e-5

    for k in range(n_tau):
        for j in range(n_pop):
            tech       = tau[k]
            population = pop[j]

            # initialize material and capital stocks
            # K_0 = make_capital(M_a0) directly — no delta_m_avg scaling
            M_a = M_a0.copy()
            K   = make_capital(M_a, parameters)

            converged = False

            for iter_num in range(max_iter):
                try:
                    soltemp, utiltemp, Y, exitflag = ecc_optimization(
                        population, tech, K, parameters
                    )
                except Exception as e:
                    warnings.warn(f'exception at tau={tech:.2f}, pop={population:.1f}: {e}')
                    utilmat[k, j]         = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    converged = True
                    break

                if exitflag == -2:
                    utilmat[k, j]         = np.nan
                    solutioncube[:, k, j] = np.nan
                    converged = True
                    break
                elif exitflag <= 0:
                    warnings.warn(
                        f'solver failed (exitflag={exitflag}) at '
                        f'tau={tech:.2f}, pop={population:.1f}, iter={iter_num}'
                    )
                    utilmat[k, j]         = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    converged = True
                    break

                # material and capital dynamics
                savings            = s * Y
                gross_add_material = recycle(M_a, M_max, savings, tech, parameters)
                dM                 = gross_add_material - delta_m * M_a
                dK                 = make_capital(gross_add_material, parameters) - delta_k * K

                # scale-independent convergence check (percent changes)
                pct_dM    = np.abs(dM) / np.maximum(M_a, 1e-10)
                pct_dK    = abs(dK) / max(K, 1e-10)
                pct_d_max = max(np.max(pct_dM), pct_dK)

                if pct_d_max < tolerance:
                    utilmat[k, j]            = utiltemp
                    solutioncube[0:11, k, j] = soltemp
                    solutioncube[11, k, j]   = K
                    converged = True
                    break

                # update stocks
                M_a = M_a + dM
                K   = K   + dK

                if iter_num == max_iter - 1:
                    warnings.warn(
                        f'did not converge for tau={tech:.2f}, pop={population:.1f}'
                    )
                    utilmat[k, j]         = np.inf
                    solutioncube[:, k, j] = np.inf

            print(f'  tau={tech:.2f}, pop={population:.0f}: '
                  f'util={-utilmat[k,j]:.4f}, K={K:.6f}, '
                  f'converged={converged}',
                  flush=True)

    # flip sign — optimizer minimizes negative utility
    utilmat = -utilmat

    # save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename  = f'ECC_solutions_{timestamp}.npz'
    np.savez_compressed(
        filename,
        solutioncube=solutioncube,
        utilmat=utilmat,
        tau=tau,
        pop=pop
    )
    print(f'results saved to {filename}')

    return utilmat, solutioncube, tau, pop, parameters


if __name__ == '__main__':
    run_ecc_simulation(
        tau=np.array([0.0]),
        pop=np.array([8.0]),
        max_iter=1000
    )
