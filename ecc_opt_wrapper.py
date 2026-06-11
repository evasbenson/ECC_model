"""
ecc_opt_wrapper.py

Translated from ECC_opt_wrapper_v6b.m

Key changes from previous version:
- Capital K now evolves via material stock dynamics (recycle + make_capital)
- Fixed-point iteration converges on dK (change in capital) not kmax/nmax
- Material stocks M_a tracked across iterations (5 categories)
- K_0 initialized from material stocks and depreciation rates
- recycle() computes gross material additions from active + waste stocks
- make_capital() converts recycled material to capital investment
- ecc_optimization now takes K directly instead of kmax/nmax
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
    rho_max     = parameters['rho_max']
    rho_current = parameters['rho_current']
    beta_max    = parameters['beta_max']
    beta_current = parameters['beta_current']
    zeta_max    = parameters['zeta_max']
    zeta_current = parameters['zeta_current']
    c3          = parameters['c3']
    delta_m     = parameters['d_m']

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

    Converts recycled material vector to a scalar capital investment.

    If phi_subs == 0: log-linear (Cobb-Douglas) aggregation
    Otherwise: CES aggregation

    Parameters
    ----------
    material : array (5,) — material inputs
    parameters : dict

    Returns
    -------
    float : capital investment (same units as K)
    """
    phi_share = parameters['phi_share']
    phi_subs  = parameters['phi_subs']

    if phi_subs == 0:
        log_investment = np.dot(phi_share, np.log(material))
        return 1000 * np.exp(log_investment)
    else:
        return 1000 * (np.dot(phi_share, material ** phi_subs)) ** (1.0 / phi_subs)


def run_ecc_simulation(tau=None, pop=None, max_iter=500):
    """
    Runs the full ECC simulation across a technology x population grid.
    Translated from ECC_opt_wrapper_v6b.m

    Fixed-point iteration now converges on dK (net change in capital stock)
    rather than kmax/nmax from the previous version.

    Parameters
    ----------
    tau : array — technology levels (default: [0] for single point test)
    pop : array — population levels in billions (default: [8])
    max_iter : int — maximum fixed-point iterations (default: 500)

    Returns
    -------
    utilmat : (len(tau), len(pop)) array — utility at each grid point
    solutioncube : (12, len(tau), len(pop)) array — solutions
    tau, pop : arrays used
    parameters : dict
    """
    parameters = get_parameters()

    lfpr    = parameters['lfpr']
    H       = parameters['H']
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

    for k in range(n_tau):
        for j in range(n_pop):
            tech       = tau[k]
            population = pop[j]
            L          = lfpr * population

            M_a = M_a0.copy()
            delta_m_avg = np.dot(M_a, delta_m) / M_a.sum()
            K_0 = delta_m_avg / delta_k * make_capital(M_a, parameters)
            K   = K_0

            if L <= 1:
                tolerance = 1e-4
            else:
                scale_factor = max(1, population / 25)
                tolerance    = 1e-5 * scale_factor

            converged = False

            for iter_num in range(max_iter):
                try:
                    soltemp, utiltemp, Y, exitflag = ecc_optimization(
                        population, tech, K, parameters
                    )
                except Exception as e:
                    warnings.warn(f'exception at tau={tech:.2f}, pop={population:.1f}: {e}')
                    utilmat[k, j]        = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    converged = True
                    break

                if exitflag == -2:
                    utilmat[k, j]        = np.nan
                    solutioncube[:, k, j] = np.nan
                    converged = True
                    break
                elif exitflag <= 0:
                    warnings.warn(
                        f'solver failed (exitflag={exitflag}) at '
                        f'tau={tech:.2f}, pop={population:.1f}, iter={iter_num}'
                    )
                    utilmat[k, j]        = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    converged = True
                    break

                savings           = s * Y
                gross_add_material = recycle(M_a, M_max, savings, tech, parameters)
                dM                = gross_add_material - delta_m * M_a
                dK                = make_capital(gross_add_material, parameters) - delta_k * K

                if abs(dK) < tolerance:
                    utilmat[k, j]           = utiltemp
                    solutioncube[0:11, k, j] = soltemp
                    solutioncube[11, k, j]   = K
                    converged = True
                    break

                M_a = M_a + dM
                K   = K   + dK

                if iter_num == max_iter - 1:
                    warnings.warn(
                        f'did not converge for tau={tech:.2f}, pop={population:.1f}'
                    )
                    utilmat[k, j]        = np.inf
                    solutioncube[:, k, j] = np.inf

            print(f'  tau={tech:.2f}, pop={population:.0f}: '
                  f'util={-utilmat[k,j]:.4f}, K={K:.4f}, '
                  f'flag={exitflag if converged else "no_conv"}',
                  flush=True)

    utilmat = -utilmat

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename  = f'ECC_solutions_{timestamp}.npz'
    np.savez_compressed(
        filename,
        solutioncube=solutioncube,
        utilmat=utilmat,
        tau=tau,
        pop=pop,
        parameters=parameters
    )
    print(f'results saved to {filename}')

    return utilmat, solutioncube, tau, pop, parameters


if __name__ == '__main__':
    run_ecc_simulation(
        tau=np.array([0.0]),
        pop=np.array([8.0]),
        max_iter=500
    )
